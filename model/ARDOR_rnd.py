# ARDOR_rnd.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RNDModule(nn.Module):
    """
    RND with (amodal_feat, action_stack) as input.

    - amodal_feat:  (B, k, 256)  or (B, k, feat_size)
    - action_stack: (B, k, 3)    # stacked (Δx, Δy, Δpitch(rad))

    input = concat([amodal_feat.flatten, action_stack.flatten])  # (B, k*feat_size + k*3)
    intrinsic = MSE( Target(x), Predictor(x) )
    """
    def __init__(
        self,
        feat_dim=(256, 6),        # (feat_size=256, feat_stack=k) - amodal_max_instance=1 → 256
        act_stack_dim=(6, 3),     # (stack=k, action_dim=3) - frame_stack=6
        hidden_dim=512,
        lr=1e-4,
        device="cuda",
        use_layernorm=True,
        # --- Normalization hyperparameters ---
        ema_alpha=1e-3,           # EMA coefficient for running statistics (smaller → slower update)
        init_mean=0.0,
        init_var=1.0,
    ):
        super().__init__()
        self.device = torch.device(device)

        # dims
        self.feat_size = int(feat_dim[0])      # 256 (amodal_max_instance=1 * 256)
        self.feat_k    = int(feat_dim[1])      # 6 (frame_stack)
        self.act_k     = int(act_stack_dim[0]) # 6 (frame_stack)
        self.act_dim   = int(act_stack_dim[1]) # 3

        # total input dim = k*256 + k*3 = 6*256 + 6*3 = 1536 + 18 = 1554
        self.input_dim = self.feat_k * self.feat_size + self.act_k * self.act_dim

        # Optional: input normalization (to mitigate scale/distribution mismatch)
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.input_ln = nn.LayerNorm(self.input_dim)

        # Target network (fixed)
        self.target_net = nn.Sequential(
            nn.Linear(self.input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, hidden_dim),     nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for p in self.target_net.parameters():
            p.requires_grad = False

        # Predictor network (trainable)
        self.predictor_net = nn.Sequential(
            nn.Linear(self.input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, hidden_dim),     nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.to(self.device)
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)

        # -------- Running statistics (EMA) --------
        # Use register_buffer so that they are saved/restored with state_dict
        self.register_buffer("r_mean", torch.tensor(float(init_mean)))
        self.register_buffer("r_var",  torch.tensor(float(init_var)))
        self.register_buffer("r_count", torch.tensor(0.0))  # for warmup / monitoring
        self.ema_alpha = float(ema_alpha)

    # -------- helpers --------
    def _flatten_input(self, amodal_feat, action_stack):
        """
        amodal_feat:  (B, k, feat_size)   e.g., (B, 6, 256)
        action_stack: (B, k, 3)
        -> (B, k*feat_size + k*3)
        """
        amodal_feat = amodal_feat.to(self.device).float()
        action_stack = action_stack.to(self.device).float()

        B, k1, fdim = amodal_feat.shape
        B2, k2, adim = action_stack.shape
        assert k1 == self.feat_k and fdim == self.feat_size, \
            f"amodal_feat shape mismatch: got {(k1, fdim)}, expected {(self.feat_k, self.feat_size)}"
        assert k2 == self.act_k and adim == self.act_dim, \
            f"action_stack shape mismatch: got {(k2, adim)}, expected {(self.act_k, self.act_dim)}"
        assert B == B2, "batch size mismatch"

        feat_flat = amodal_feat.reshape(B, -1)   # (B, k*feat_size)
        act_flat  = action_stack.reshape(B, -1)  # (B, k*3)
        x = torch.cat([feat_flat, act_flat], dim=1)  # (B, input_dim)
        if self.use_layernorm:
            x = self.input_ln(x)
        return x

    # -------- forward --------
    def forward_target(self, amodal_feat, action_stack):
        x = self._flatten_input(amodal_feat, action_stack)
        return self.target_net(x)

    def forward_predictor(self, amodal_feat, action_stack):
        x = self._flatten_input(amodal_feat, action_stack)
        return self.predictor_net(x)

    # -------- Running-stats update (EMA) --------
    @torch.no_grad()
    def _update_running_stats(self, batch_vals: torch.Tensor):
        """
        batch_vals: (B,) tensor of raw intrinsic (MSE)
        Update running mean/variance using EMA (variance tracked directly).
        """
        if batch_vals.numel() == 0:
            return

        b_mean = batch_vals.mean()
        # Add small epsilon for numerical stability
        b_var = batch_vals.var(unbiased=False) + 1e-6

        a = self.ema_alpha
        # Count can be used as a simple indicator for warmup
        self.r_count += 1.0
        # EMA update
        self.r_mean = (1 - a) * self.r_mean + a * b_mean
        self.r_var  = (1 - a) * self.r_var  + a * b_var

    # -------- Public API --------
    @torch.no_grad()
    def compute_intrinsic_reward(
        self,
        amodal_feat,
        action_stack,
        normalize: bool = True,
        clip: float = 5.0,
        relu: bool = True,        # remove negative values (keep exploration bonus ≥ 0)
        update_stats: bool = True # set False when only evaluating/logging
    ):
        """
        Returns:
            (B,) — intrinsic reward

        Default behavior:
          - Compute RND MSE
          - Normalize with running mean/variance (z-score)
          - Optionally apply ReLU and clipping
        """
        t = self.forward_target(amodal_feat, action_stack)
        p = self.forward_predictor(amodal_feat, action_stack)
        mse = F.mse_loss(p, t, reduction='none').mean(dim=1)  # (B,)

        if not normalize:
            return mse  # use raw values (not recommended in most cases)

        # Update running statistics only during training
        if update_stats:
            self._update_running_stats(mse.detach())

        # z-score: (x - mean) / std
        std = torch.sqrt(self.r_var.clamp_min(1e-8))
        z = (mse - self.r_mean) / std

        # Optionally remove negative values (exploration bonus is usually non-negative)
        if relu:
            z = torch.relu(z)

        # Optional clipping to avoid exploding values and stabilize scale
        if clip is not None:
            z = torch.clamp(z, max=clip)

        return z

    def update(self, amodal_feat, action_stack):
        """
        Off-policy update for the predictor network.
        Typically called with next-state tuples from replay buffer.

        Note:
            Running statistics are NOT updated here; use compute_intrinsic_reward
            with update_stats=True during training.
        """
        t = self.forward_target(amodal_feat, action_stack)
        p = self.forward_predictor(amodal_feat, action_stack)
        loss = F.mse_loss(p, t, reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())
