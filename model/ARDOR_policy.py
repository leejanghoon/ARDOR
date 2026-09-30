# ARDOR_policy.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
import math

# ---------------------
#  Actor
# ---------------------
class Actor(nn.Module):
    
    def __init__(self, feat_dim, action_stack_dim, action_dim):
        
        super().__init__()

        self.feat_size = feat_dim[0]     
        self.feat_k = feat_dim[1]      
        in_feat = self.feat_size * self.feat_k 

        self.ln_amodal = nn.LayerNorm(in_feat)
        self.amodal_net = nn.Sequential(
            nn.Linear(in_feat, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )


        self.act_stack_k = action_stack_dim[0]   
        self.single_act_dim = action_stack_dim[1] 
        in_action = self.act_stack_k * self.single_act_dim  

        self.ln_act = nn.LayerNorm(in_action)

        self.action_stack_net = nn.Sequential(
            nn.Linear(in_action, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.cand_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        # context + candidate → score
        self.score_head = nn.Sequential(
            nn.Linear(256 + 128 + 64, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


    def forward(self, amodal_feat, action_stack, cand_actions):
        """
        return logits over candidates: (B, K)
        """
        B, K, _ = cand_actions.shape

        # context
        amodal_flat = self.ln_amodal(amodal_feat.view(B, -1))
        amodal_emb  = self.amodal_net(amodal_flat)           # (B,256)

        act_flat    = self.ln_act(action_stack.view(B, -1))
        action_emb  = self.action_stack_net(act_flat)        # (B,128)

        ctx = torch.cat([amodal_emb, action_emb], dim=1)     # (B,384)

        # candidates
        c_flat = cand_actions.reshape(B * K, 3)              # (B*K,3)
        c_emb  = self.cand_mlp(c_flat)                       # (B*K,64)

        ctx_rep = ctx.repeat_interleave(K, dim=0)            # (B*K,384)
        x = torch.cat([ctx_rep, c_emb], dim=1)               # (B*K,448)
        logits = self.score_head(x).view(B, K)               # (B,K)
        return logits

# ---------------------
#  Critic
# ---------------------
class Critic(nn.Module):
    """
    Q((amodal_feat, action_stack), current_action) -> Q-value
    """
    def __init__(self, feat_dim, action_stack_dim, action_dim):

        super().__init__()

        # amodal_feat part
        self.feat_size = feat_dim[0]
        self.feat_k = feat_dim[1]
        in_feat = self.feat_size * self.feat_k

        # linear normalization in amodal feature
        self.ln_amodal = nn.LayerNorm(in_feat)

        self.amodal_net = nn.Sequential(
            nn.Linear(in_feat, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # action_stack part
        self.act_stack_k = action_stack_dim[0]
        self.single_act_dim = action_stack_dim[1]
        in_action_stack = self.act_stack_k * self.single_act_dim

        # linear normalization in amodal feature
        self.ln_act = nn.LayerNorm(in_action_stack)

        self.action_stack_net = nn.Sequential(
            nn.Linear(in_action_stack, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.final_net = nn.Sequential(
            nn.Linear(256 + 128 + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
  
    def forward(self, amodal_feat, action_stack, action):
   
        B, k, fdim = amodal_feat.shape
        amodal_flat = amodal_feat.view(B, -1)
        amodal_flat = self.ln_amodal(amodal_flat) # LN
        amodal_emb = self.amodal_net(amodal_flat)

        B2, k2, adim = action_stack.shape
        act_stack_flat = action_stack.view(B2, -1)
        act_stack_flat = self.ln_act(act_stack_flat) # LN
        action_emb = self.action_stack_net(act_stack_flat)

        x = torch.cat([amodal_emb, action_emb, action], dim=1)  # (B,256+128+6)
        q_val = self.final_net(x)                                       # (B,1)
        return q_val
    
    
class ReplayBuffer:
    """
    Priority sampling with *symmetric softmax on raw rewards*.

    - p_i ∝ softmax( beta * z_i ), where z_i = (r_i - mean(r)) / std(r)
    - Optional smoothing:
        * rho:      mix with uniform -> p <- (1-rho)*p + rho*(1/N)
        * eps_floor: per-item minimum floor before final renorm
        * max_prob: hard cap per item (apply then renormalize)

    sampler_cfg keys used:
      - uniform_ratio (float in [0,1], default 0.3)
      - beta         (float, default 1.0)
      - rho          (float in [0,1], default 0.0)
      - eps_floor    (float >= 0, default 0.0)
      - max_prob     (float in (0,1], optional)

    Stored tuple per transition:
      (s, s_amodal, a, r, s2, s2_amodal, d, cand, idx, cand2)
    """

    def __init__(self, capacity=100000):
        from collections import deque
        self.buffer = deque(maxlen=capacity)

        # --- probability cache ---
        self._dirty = True                 # buffer changed since last freeze
        self._cached_p = None              # cached prob. vector (np.float64, sum=1)
        self._cached_cfg = None            # cfg snapshot used to build cached p
        self._cached_N = 0                 # buffer size when cached p was built

    def __len__(self):
        return len(self.buffer)

    # ---------------------------
    # Gather utilities
    # ---------------------------
    def _gather_by_indices(self, indices, device="cuda"):
        """
        Return batched tensors from selected indices.
        Expected shapes per item:
          s, s2:           (k, 3)
          s_amodal, s2_amodal: (D_flat,)
          a:               (3,)
          r, d:            scalar
          cand, cand2:     (K,3), (K2,3)
          idx:             scalar(int)
        """
        s_list, sA_list, a_list, r_list, s2_list, sA2_list, d_list, cand_list, idx_list, cand2_list = ([] for _ in range(10))

        for i in indices:
            s, sA, a, r, s2, sA2, d, cand, idx, cand2 = self.buffer[int(i)]
            s_list.append(np.asarray(s,   dtype=np.float32))
            sA_list.append(np.asarray(sA, dtype=np.float32))
            a_list.append(np.asarray(a,   dtype=np.float32))
            r_list.append(np.float32(r))
            s2_list.append(np.asarray(s2,  dtype=np.float32))
            sA2_list.append(np.asarray(sA2,dtype=np.float32))
            d_list.append(np.float32(d))
            cand_list.append(np.asarray(cand,  dtype=np.float32))
            idx_list.append(np.int64(idx))
            cand2_list.append(np.asarray(cand2, dtype=np.float32))

        def t(x, dtype=torch.float32):
            return torch.as_tensor(np.stack(x, axis=0), dtype=dtype, device=device)

        s    = t(s_list)                               # (B,k,3)
        sA   = t(sA_list)                              # (B,D_flat)
        a    = t(a_list)                               # (B,3)
        r    = t(r_list).unsqueeze(1)                  # (B,1)
        s2   = t(s2_list)                              # (B,k,3)
        sA2  = t(sA2_list)                             # (B,D_flat)
        d    = t(d_list).unsqueeze(1)                  # (B,1)
        cand = t(cand_list)                            # (B,K,3)
        idx  = torch.as_tensor(np.stack(idx_list), dtype=torch.long, device=device)  # (B,)
        cand2= t(cand2_list)                           # (B,K2,3)
        return s, sA, a, r, s2, sA2, d, cand, idx, cand2

    def push(self, s, s_amodal, a, r, s2, s2_amodal, d, cand=None, idx=None, cand2=None):
        self.buffer.append((s, s_amodal, a, r, s2, s2_amodal, d, cand, idx, cand2))
        self._dirty = True   # buffer mutated -> invalidate cache

    # ---------------------------
    # Probability utilities
    # ---------------------------
    @staticmethod
    def _normalize_prob_safe(p: np.ndarray) -> np.ndarray:
        """
        Clean NaN/Inf/negatives and make the sum exactly 1.
        Fallback to uniform when needed.
        """
        p = np.asarray(p, dtype=np.float64)
        bad = ~np.isfinite(p)
        if bad.any():
            p[bad] = 0.0
        p = np.maximum(p, 0.0)

        s = p.sum()
        if not np.isfinite(s) or s <= 0.0:
            N = p.shape[0]
            return np.full(N, 1.0 / N, dtype=np.float64)

        p /= s
        # force exact sum=1
        residual = 1.0 - p.sum()
        if abs(residual) > 1e-12:
            p[-1] = max(0.0, p[-1] + residual)
            s2 = p.sum()
            if not np.isfinite(s2) or s2 <= 0.0:
                N = p.shape[0]
                return np.full(N, 1.0 / N, dtype=np.float64)
            p /= s2
        return p

    @staticmethod
    def _softmax_from_rewards(rewards: np.ndarray, cfg: dict) -> np.ndarray:
        """
        Symmetric softmax on raw rewards with z-score:
          z = (r - mean)/std
          p = softmax(beta * z)
          p <- (1-rho)*p + rho*(1/N)
          p <- max(p, eps_floor)
          p <- min(p, max_prob)  # optional
          p <- normalize

        Keys:
          beta (float), rho (float), eps_floor (float), max_prob (float)
        """
        beta = float(cfg.get("beta", 1.0))
        rho  = float(cfg.get("rho", 0.0))
        eps_floor = float(cfg.get("eps_floor", 0.0))
        max_prob_cap = cfg.get("max_prob", None)

        r = np.asarray(rewards, dtype=np.float64)
        # z-score for scale invariance
        mu = r.mean()
        sd = r.std()
        z = (r - mu) / (sd + 1e-8)

        # stable softmax
        logits = beta * z
        logits -= logits.max()
        logits = np.clip(logits, -745.0, 80.0)   # exp under/overflow guard
        p = np.exp(logits)
        s = p.sum()
        p = p / s if s > 0.0 else np.full_like(p, 1.0 / len(p))

        # mix with uniform to reduce peaky distributions (optional)
        if rho > 0.0:
            N = len(p)
            p = (1.0 - rho) * p + rho * (1.0 / N)

        # per-item floor (optional)
        if eps_floor > 0.0:
            p = np.maximum(p, eps_floor)

        p = ReplayBuffer._normalize_prob_safe(p)

        # hard cap (optional)
        if isinstance(max_prob_cap, (float, int)) and max_prob_cap > 0.0:
            p = np.minimum(p, float(max_prob_cap))
            p = ReplayBuffer._normalize_prob_safe(p)

        return p

    # ---------------------------
    # Cache builder (Softmax probs)
    # ---------------------------
    def freeze_distribution(self, cfg: dict):
        """
        Call once right after an episode ends.
        Build and cache softmax-based probabilities from current rewards.
        """
        N = len(self.buffer)
        if N == 0:
            self._cached_p = None
            self._cached_cfg = dict(cfg)
            self._cached_N = 0
            self._dirty = False
            return

        rewards = np.fromiter((t[3] for t in self.buffer), dtype=np.float64, count=N)
        p = self._softmax_from_rewards(rewards, cfg)
        p = self._normalize_prob_safe(p)  # final guard

        self._cached_p = p.astype(np.float64, copy=False)
        self._cached_cfg = dict(cfg)
        self._cached_N = N
        self._dirty = False

    # ---------------------------
    # Sampling
    # ---------------------------
    def sample(self, batch_size, device="cuda", sampler_cfg=None):
        """
        If cache is valid, use it (fast path).
        Otherwise, build probs on the fly from current rewards.
        """
        def _cfg_keys_ok(cached, given):
            keys = ["beta", "rho", "eps_floor", "uniform_ratio", "max_prob"]
            for k in keys:
                if cached.get(k) != given.get(k):
                    return False
            return True

        # --- try cached fast path ---
        use_cache = (
            (self._cached_p is not None) and
            (not self._dirty) and
            (self._cached_N == len(self.buffer)) and
            (sampler_cfg is not None) and
            (self._cached_cfg is not None) and
            _cfg_keys_ok(self._cached_cfg, sampler_cfg)
        )

        if use_cache:
            cfg = self._cached_cfg
            uniform_ratio = float(cfg.get("uniform_ratio", 0.3))
            n_weighted = int(round(batch_size * (1.0 - uniform_ratio)))
            n_uniform  = batch_size - n_weighted
            N = self._cached_N

            # weighted
            replace_w = (n_weighted > N)
            if n_weighted > 0:
                p = self._normalize_prob_safe(self._cached_p)  # guard
                weighted_idx = np.random.choice(N, size=min(n_weighted, N),
                                                replace=replace_w, p=p)
            else:
                weighted_idx = np.empty((0,), dtype=np.int64)

            # uniform (avoid overlap with weighted where possible)
            if n_uniform > 0:
                taken = np.zeros(N, dtype=bool); taken[weighted_idx] = True
                pool = np.nonzero(~taken)[0]
                if len(pool) >= n_uniform:
                    uniform_idx = np.random.choice(pool, size=n_uniform, replace=False)
                else:
                    uniform_idx = np.random.choice(np.arange(N), size=n_uniform, replace=True)
            else:
                uniform_idx = np.empty((0,), dtype=np.int64)

            indices = np.concatenate([weighted_idx, uniform_idx])
            np.random.shuffle(indices)
            return self._gather_by_indices(indices, device=device)

        # --- on-the-fly path ---
        sampler_cfg = sampler_cfg or {}
        uniform_ratio = float(sampler_cfg.get("uniform_ratio", 0.3))

        N = len(self.buffer)
        assert N > 0, "ReplayBuffer is empty"

        n_weighted = int(round(batch_size * (1.0 - uniform_ratio)))
        n_uniform  = batch_size - n_weighted

        rewards = np.fromiter((t[3] for t in self.buffer), dtype=np.float64, count=N)
        p = self._softmax_from_rewards(rewards, sampler_cfg)
        p = self._normalize_prob_safe(p)  # guard

        # weighted
        if n_weighted > 0:
            weighted_idx = np.random.choice(N, size=min(n_weighted, N),
                                            replace=(n_weighted > N), p=p)
        else:
            weighted_idx = np.array([], dtype=np.int64)

        # uniform (fast set-diff)
        if n_uniform > 0:
            taken = np.zeros(N, dtype=bool)
            if len(weighted_idx) > 0:
                taken[weighted_idx] = True
            pool = np.nonzero(~taken)[0]
            if len(pool) >= n_uniform:
                uniform_idx = np.random.choice(pool, size=n_uniform, replace=False)
            else:
                uniform_idx = np.random.choice(np.arange(N), size=n_uniform, replace=True)
        else:
            uniform_idx = np.array([], dtype=np.int64)

        indices = np.concatenate([weighted_idx, uniform_idx])
        np.random.shuffle(indices)
        return self._gather_by_indices(indices, device=device)




# ---------------------
#  ARDORPolicy
# ---------------------
class ARDORPolicy:
    def __init__(self, actor, critic, model_cfg,
                 actor_lr=1e-4, critic_lr=1e-4,
                 gamma=0.99, tau=0.005, device="cuda"):
        self.device = torch.device(device)

        self.model_cfg = model_cfg
        self.gamma = gamma
        self.tau = tau

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Target critic only (DQN-style backup over candidate actions)
        state_dim = self.model_cfg['state_dim']
        action_stack_dim = self.model_cfg['action_stack_dim']
        action_dim = self.model_cfg["action_dim"]      # should be 3
        self.target_critic = Critic(state_dim, action_stack_dim, action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Candidate-selection settings
        self.K = self.model_cfg.get('candidate_K', 5)
        self.selector_tau = self.model_cfg.get('selector_tau', 1.0)          # softmax temperature for distillation
        self.selector_ce_weight = self.model_cfg.get('selector_ce_weight', 0.0)  # optional CE(idx) weight

        # Epsilon-greedy exploration over candidates
        self.eps = self.model_cfg.get('epsilon_start', 0.2)
        self.eps_end = self.model_cfg.get('epsilon_end', 0.05)
        self.eps_decay = self.model_cfg.get('epsilon_decay', 0.999)

        # Replay sampling configuration (priority + uniform mix)
        self.replay_sampling = self.model_cfg.get("replay_sampling", {
            "uniform_ratio": 0.3,   # 70% priority + 30% uniform mix
            "beta": 0.7,            # softmax temperature (↑ sharper, ↓ flatter)
            "rho": 0.1,             # mix with uniform to reduce over-concentration (0.0–0.2 recommended)
            "eps_floor": 0.0,       # minimum probability floor per item (typically 0.0–1e-6)
            "max_prob": 0.1         # optional cap on per-transition max probability (None to disable)
        })  # limits how much probability a single sample can take (to avoid over-focus)

    @torch.no_grad()
    def _sample_candidates(self, env, env_id):
        """Sample K*P candidate actions from the environment for a given env_id."""
        K  = self.K
        P  = self.model_cfg.get('pitch_P', 5)
        cand_np = env.sample_candidates_xy(K=K, P=P, env_id=env_id)  # (K*P, 3) = [dx, dy, d_pitch]
        return torch.tensor(cand_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, K*P, 3)

    def select_action(self, amodal_feat, action_stack, env, env_id=0):
        """
        Generate candidates → run actor to get logits → pick one via ε-greedy over candidates.

        Returns:
            action:     (1, 3)
            cand_t:     (1, K, 3)  candidate list
            idx_t:      (1,) index of the chosen candidate
        """
        cand_t = self._sample_candidates(env, env_id)                # (1, K, 3)
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(amodal_feat, action_stack, cand_t)   # (1, K)
            if np.random.rand() < self.eps:
                idx = np.random.randint(0, logits.shape[1])
            else:
                idx = int(torch.argmax(logits, dim=1).item())
        self.actor.train()
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        act = cand_t[0, idx]                                         # (3,)
        idx_t = torch.tensor([idx], dtype=torch.long, device=self.device)
        return act.unsqueeze(0), cand_t, idx_t

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < 250:
            return 0.0, 0.0, {
                "Q_mean": 0.0, "y_mean": 0.0, "y_std": 0.0,
                "p_star_entropy": 0.0, "actor_entropy": 0.0
            }

        s, s_amodal, a, r, s2, s2_amodal, d, cand, idx, cand2 = replay_buffer.sample(
            batch_size, device=self.device, sampler_cfg=self.replay_sampling
        )

        # -------------------
        # Critic update (DQN-style backup over next-step candidates)
        # -------------------
        with torch.no_grad():
            if cand2.shape[1] > 0:
                B, K2, _ = cand2.shape
                c2_flat = cand2.view(B * K2, 3)
                sA2_rep = s2_amodal.repeat_interleave(K2, dim=0)
                s2_rep  = s2.repeat_interleave(K2, dim=0)

                # 1) Select a* using the online critic
                q_next_online = self.critic(sA2_rep, s2_rep, c2_flat).view(B, K2)   # (B, K2)
                a_star = q_next_online.argmax(dim=1, keepdim=True)                 # (B, 1)

                # 2) Evaluate that a* with target_critic
                q_next_targ = self.target_critic(sA2_rep, s2_rep, c2_flat).view(B, K2)  # (B, K2)
                Q_next = q_next_targ.gather(1, a_star)                                  # (B, 1)
            else:
                Q_next = torch.zeros_like(r)

            y = r + self.gamma * (1.0 - d) * Q_next

        Q = self.critic(s_amodal, s, a)
        critic_loss = F.smooth_l1_loss(Q, y, beta=1.0)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        # -------------------
        # Actor update (distill policy to Q(s, ·) over current candidates)
        # -------------------
        if cand.shape[1] > 0:
            logits = self.actor(s_amodal, s, cand)   # (B, K)
            valid = (idx >= 0)

            if valid.any():
                sA_v, s_v, cand_v = s_amodal[valid], s[valid], cand[valid]
                Bv, K, _ = cand_v.shape

                c_flat = cand_v.view(Bv * K, 3)
                sA_rep = sA_v.repeat_interleave(K, dim=0)
                s_rep  = s_v.repeat_interleave(K, dim=0)

                with torch.no_grad():
                    qk = self.critic(sA_rep, s_rep, c_flat).view(Bv, K)           # (Bv, K)
                    p_star = torch.softmax(qk / self.selector_tau, dim=1)        # soft target over candidates

                log_p = torch.log_softmax(logits[valid], dim=1)
                kl = torch.nn.KLDivLoss(reduction='batchmean')(log_p, p_star)

                if self.selector_ce_weight > 0.0:
                    ce = F.cross_entropy(logits[valid], idx[valid].clamp_min(0))
                    actor_loss = kl + self.selector_ce_weight * ce
                else:
                    actor_loss = kl
            else:
                actor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            actor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # -------------------
        # Soft update (critic target)
        # -------------------
        self._soft_update(self.critic, self.target_critic)

        with torch.no_grad():
            Q_mean = float(Q.mean().item())
            y_mean = float(y.mean().item())
            y_std  = float(y.std().item())

            p_star_entropy = float('nan')
            actor_entropy  = float('nan')

            if 'logits' in locals() and cand.shape[1] > 0:
                if 'valid' in locals() and valid.any():
                    probs_v = logits[valid].softmax(dim=1)
                    actor_entropy = float(
                        (-(probs_v * (probs_v.clamp_min(1e-12)).log()).sum(dim=1)).mean().item()
                    )
                else:
                    probs = logits.softmax(dim=1)
                    actor_entropy = float(
                        (-(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)).mean().item()
                    )

            if 'p_star' in locals():
                p_star_entropy = float(
                    (-(p_star * (p_star.clamp_min(1e-12)).log()).sum(dim=1)).mean().item()
                )

            diag = {
                "Q_mean": Q_mean,
                "y_mean": y_mean,
                "y_std":  y_std,
                "p_star_entropy": p_star_entropy,
                "actor_entropy":  actor_entropy,
            }

        return float(critic_loss.item()), float(actor_loss.item()), diag

    def _soft_update(self, net, net_target):
        """Soft-update target network parameters."""
        for p, tp in zip(net.parameters(), net_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # Kept for compatibility with older training loops
    def reset_noise(self):
        return

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint.get("target_critic", checkpoint["critic"]))
