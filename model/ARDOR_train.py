# ARDOR_train.py
import argparse
import os
import pickle
import numpy as np
import torch
import time
import genesis as gs
import sys
from _3D_recon.fusion import meshwrite_glb
from model.ARDOR_env import ARDOR_Env
from model.ARDOR_policy import Actor, Critic, ARDORPolicy, ReplayBuffer
from model.ARDOR_rnd import RNDModule
from torch.utils.tensorboard import SummaryWriter

import cv2
import random
from dataclasses import dataclass
import csv
import math

# --------- Redistribution / training hyperparameters ---------
SPIKE_DELTA_CR   = 0.10     # CR spike threshold (replaces old SPIKE_DELTA)
SPIKE_DELTA_OCC  = 0.05     # OCC spike threshold (recommend separate scale)
NEAR_LAMBDA_CR   = 0.10     # Kernel lambda for CR redistribution
NEAR_LAMBDA_OCC  = 0.10     # Kernel lambda for OCC redistribution
WEIGHT_MODE      = "exp"    # "exp" | "linear" | "uniform" | "poly" | "mix"
POLY_P           = 1.0
MIX_ETA          = 0.4

# When there are multiple spikes: mixing ratio between
# "emphasize only around the last spike" vs "emphasize around all spikes".
# 0.0 → only last spike, 1.0 → sum of all spikes.
MULTI_SPIKE_ETA  = 0.5

UPDATE_RATIO = 0.8       # Number of update steps ≈ UPDATE_RATIO × episode length
MIN_BUFFER   = 1000      # Minimum buffer size before starting updates

# ----- Hampel correction hyperparameters -----
HAMPEL_WINDOW    = 7      # Window size for median/MAD
HAMPEL_N_SIGMA   = 3.0    # Threshold multiplier for spike detection
HAMPEL_HOLD_LEN  = 4      # Hold length to preserve true step-like increases
SUPPRESS_RADIUS  = 2      # Radius to suppress fake spikes near corrected indices


@dataclass
class StepRec:
    s: np.ndarray;  sA: np.ndarray;  a: np.ndarray
    r_base: float;  r3d: float;  r_occ: float   # explicit decomposition for redistribution
    s2: np.ndarray; sA2: np.ndarray; d: float
    cand: np.ndarray; idx: int; cand2: np.ndarray
    t: int

# --------- Transition weighting utilities ---------
def make_weights_single(t_spike, steps, mode="exp", near_lambda=0.25, poly_p=1.0, mix_eta=0.3):
    steps_arr = np.asarray(steps, dtype=np.float32)
    mask = steps_arr <= t_spike             # zero out steps after the spike
    d = (t_spike - steps_arr)[mask]         # distance to spike for previous steps
    if mode == "exp":
        w = np.exp(-near_lambda * d)
    elif mode == "linear":
        w = (d.max() + 1.0) - d
        w = np.clip(w, 1e-6, None)
    elif mode == "uniform":
        w = np.ones_like(d)
    elif mode == "poly":
        w = (d + 1.0) ** poly_p
        w = w.max() + 1.0 - w
        w = np.clip(w, 1e-6, None)
    elif mode == "mix":
        w_exp = np.exp(-near_lambda * d)
        w_uni = np.ones_like(d)
        w = (1.0 - mix_eta) * w_exp + mix_eta * w_uni
    else:
        w = np.exp(-near_lambda * d)
    w /= (w.sum() + 1e-8)
    out = np.zeros_like(steps_arr, dtype=np.float32)
    out[mask] = w
    return out

def make_weights_multi(spikes, steps, mode="exp",
                       near_lambda=0.25, poly_p=1.0, mix_eta=0.3, eta_multi=0.5):
    """
    spikes: [t0, t1, ...] (assumed sorted ascending)
    eta_multi: 0 → use only last spike, 1 → sum over all spikes
    """
    if len(spikes) == 0:
        w = np.ones(len(steps), dtype=np.float32) / max(1, len(steps))
        return w

    # Weights centered at the last spike
    w_last = make_weights_single(spikes[-1], steps, mode, near_lambda, poly_p, mix_eta)

    # Sum of weights centered at all spikes
    w_all = np.zeros_like(w_last)
    for t in spikes:
        w_all += make_weights_single(t, steps, mode, near_lambda, poly_p, mix_eta)
    if w_all.sum() > 0:
        w_all /= (w_all.sum() + 1e-8)

    # Mix last-spike-only vs all-spikes
    w = (1.0 - eta_multi) * w_last + eta_multi * w_all
    w /= (w.sum() + 1e-8)
    return w

# ----------------- Hampel filter utilities -----------------
def _hampel_filter_preserve_steps(x, window_size=7, n_sigma=3.0, hold_len=3):
    """
    Hampel filter that tries to preserve true step-like changes.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = x_arr.copy()
    corrected = []
    N = len(x_arr)
    k = int(max(1, window_size))
    for i in range(N):
        lo = max(0, i - k)
        hi = min(N, i + k + 1)
        win = x_arr[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med)) + 1e-8
        thresh = n_sigma * 1.4826 * mad
        if np.abs(x_arr[i] - med) > thresh:
            future_hi = min(N, i + 1 + hold_len)
            future = x_arr[i:future_hi]
            if len(future) >= hold_len:
                if np.all(np.abs(future - x_arr[i]) <= 0.5 * thresh):
                    continue
            y_arr[i] = float(med)
            corrected.append(i)
    return y_arr.tolist(), corrected

def _suppress_spikes_near(spikes, corrected_idx, radius=2):
    """
    Remove spikes that lie too close to Hampel-corrected indices,
    to avoid counting artifacts created around corrections.
    """
    if not spikes or not corrected_idx:
        return spikes
    out = []
    for s in spikes:
        if all(abs(s - c) > radius for c in corrected_idx):
            out.append(s)
    return out

def analyze_sampling_distribution(replay_buffer, sampler_cfg, writer=None, step=None, log_dir=None, tag="Replay"):
    """
    Analyze and log the sampling probability distribution of the replay buffer.
    - Recommended to call after freeze_distribution.
    """
    # Ensure cached distribution (recompute if needed)
    if getattr(replay_buffer, "_dirty", True) or getattr(replay_buffer, "_cached_p", None) is None:
        replay_buffer.freeze_distribution(sampler_cfg)

    N = len(replay_buffer)
    p = getattr(replay_buffer, "_cached_p", None)
    if N == 0 or p is None:
        print("[Sampling] buffer empty or probs not cached")
        return

    p = np.asarray(p, dtype=np.float64).copy()
    rewards = np.fromiter((t[3] for t in replay_buffer.buffer), dtype=np.float64, count=N)

    # Basic statistics
    idx_top = int(np.argmax(p)); idx_bot = int(np.argmin(p))
    p_top, p_bot = float(p[idx_top]), float(p[idx_bot])
    r_top, r_bot = float(rewards[idx_top]), float(rewards[idx_bot])
    r_mu, r_sd = float(rewards.mean()), float(rewards.std() + 1e-8)
    z_top, z_bot = (r_top - r_mu) / r_sd, (r_bot - r_mu) / r_sd

    ratio_pb = p_top / max(p_bot, 1e-12)
    diff_pb  = p_top - p_bot
    diff_rb  = r_top - r_bot

    # Distribution-level metrics
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    ess = float(1.0 / (p**2).sum())
    # Gini coefficient (concentration of the probability mass)
    p_sorted = np.sort(p)
    cum = np.cumsum(p_sorted)
    gini = float(1.0 - 2.0 * (cum.sum() / (N * cum[-1])))

    # Correlation (Pearson and Spearman approximation)
    if N > 1:
        corr_r_p = float(np.corrcoef(rewards, p)[0, 1])
        rank_r = np.argsort(np.argsort(rewards))
        rank_p = np.argsort(np.argsort(p))
        corr_spearman = float(np.corrcoef(rank_r, rank_p)[0, 1])
    else:
        corr_r_p = float("nan")
        corr_spearman = float("nan")

    # Compare with ideal softmax ratio (ignoring rho, floor, cap)
    beta = float(sampler_cfg.get("beta", 1.0)) if sampler_cfg is not None else 1.0
    pred_ratio_soft = math.exp(beta * (z_top - z_bot))

    # Console summary
    print(f"[Sampling] N={N} | p_top={p_top:.6f} (r_top={r_top:.3f}, z={z_top:.2f}) "
          f"| p_bot={p_bot:.6f} (r_bot={r_bot:.3f}, z={z_bot:.2f})")
    print(f"[Sampling] p_top/p_bot={ratio_pb:.1f} | Δp={diff_pb:.6f} | Δr={diff_rb:.3f} "
          f"| H={entropy:.3f} | ESS≈{ess:.1f} | Gini={gini:.3f} "
          f"| ρ(r,p)={corr_r_p:.3f} | ρ_s≈{corr_spearman:.3f} | softmax_ratio≈{pred_ratio_soft:.1f}")

    # TensorBoard logs
    if writer is not None and step is not None:
        writer.add_scalar(f"{tag}/p_top", p_top, step)
        writer.add_scalar(f"{tag}/p_bot", p_bot, step)
        writer.add_scalar(f"{tag}/p_ratio_top_bot", ratio_pb, step)
        writer.add_scalar(f"{tag}/r_top", r_top, step)
        writer.add_scalar(f"{tag}/r_bot", r_bot, step)
        writer.add_scalar(f"{tag}/r_diff_top_bot", diff_rb, step)
        writer.add_scalar(f"{tag}/entropy", entropy, step)
        writer.add_scalar(f"{tag}/ESS", ess, step)
        writer.add_scalar(f"{tag}/Gini", gini, step)
        writer.add_scalar(f"{tag}/corr_pearson_r_p", corr_r_p, step)
        writer.add_scalar(f"{tag}/corr_spearman_r_p", corr_spearman, step)
        try:
            from torch.utils.tensorboard import SummaryWriter  # for type hint only
            writer.add_histogram(f"{tag}/p_hist", p, step)
            writer.add_histogram(f"{tag}/reward_hist", rewards, step)
        except Exception:
            pass

    # Optional CSV dump (for later offline visualization)
    if log_dir is not None:
        csv_dir = os.path.join(log_dir, "sampling_csv")
        os.makedirs(csv_dir, exist_ok=True)
        # Per-episode file name (if step is None, save a final snapshot)
        fname = f"sampling_stats_ep{step}.csv" if step is not None else "sampling_stats_final.csv"
        csv_path = os.path.join(csv_dir, fname)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "reward", "probability"])
            for i in range(N):
                w.writerow([i, float(rewards[i]), float(p[i])])



#------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="final_PER_no_near_target")
    parser.add_argument("--num_episodes", type=int, default=4000)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--epi_cnt", type=int, default=50)
    parser.add_argument("--scene", type=int)
    parser.add_argument("--depth_noise", type=bool, default=False)
    args = parser.parse_args()

    SEED = 1
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = f"cuda:{args.device}"
    print(f"Device: {device}")

    # ----------------- Config -----------------
    env_cfg = {
        "dt": 0.1,
        "episode_max_steps": 100,
        "epi_per_scene_house": int(args.epi_cnt),
        "distance_range": [6, 25],
        "pitch_range": [0, 50],
        "distance_scale": [-2, 2],
        "pitch_scale": [-20, 20],
        "init_pose": [-135, 0.4, 25, 0, 0, 0.4],
        "debug": True,
        "scene": args.scene,
        "depth_noise": args.depth_noise,
        "debug_imshow": False,
    }
    obs_cfg = {
        "img_width": 84,
        "img_height": 84,
        "frame_stack": 2,
        "fov": 60.0,
    }
    reward_cfg = {
        "intrinsic_reward_weight": 5.0,
        "_3d_recon": 15.0,
        "CR_threshold": 0.95,
        "CR_done_reward": 20,
        "step_cost": -0.2,
        "occ_ratio_reward_weight": 0.5,
        "target_visibility": 0.25,
        "near_target": 0.0,
    }
    model_cfg = {
        "state_dim": (256, obs_cfg["frame_stack"]),
        "action_stack_dim": (obs_cfg["frame_stack"], 3),
        "action_dim": 3,
        "candidate_K": 10,
        "pitch_P": 5,
        "epsilon_start": 0.2,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.999,
        "selector_tau": 1.0,
        "selector_ce_weight": 0.0,
        
        "replay_sampling": {
            "uniform_ratio": 0.3,  # mix of 70% priority + 30% uniform
            "beta": 0.7,           # softmax temperature (↑ sharper, ↓ flatter)
            "rho": 0.1,            # mixture with uniform to avoid over-concentration (0.0–0.2 recommended)
            "eps_floor": 0.0,      # minimum probability floor per sample (e.g., 0.0–1e-6)
            "max_prob": 0.1        # cap for per-transition max probability; set None to disable
        },
    }
    amodal_cfg = {
        "amodal": True,
        "config_file": "/uoais/configs/amodal_config.yaml",
        "confidence_threshold": 0.5,
        "use_cgnet": False,
        "cgnet_weight_path": "/uoais/foreground_segmentation/rgbd_fg.pth",
        "model_path": "train.py",
        "device": device
    }

    log_dir = f"logs/{args.exp_name}_{args.scene}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Save configs
    with open(os.path.join(log_dir, "cfg.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, model_cfg, amodal_cfg], f)

    # ----------------- Env/Model -----------------
    env = ARDOR_Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        amodal_cfg=amodal_cfg,
        reward_cfg=reward_cfg,
        model_cfg=model_cfg,
        show_viewer=args.viewer,
        render=args.render,
        device=device
    )

    state_dim = model_cfg['state_dim']
    action_dim = model_cfg["action_dim"]
    action_stack_dim = model_cfg['action_stack_dim']

    actor  = Actor(state_dim, action_stack_dim, action_dim)
    critic = Critic(state_dim, action_stack_dim, action_dim)
    ARDOR   = ARDORPolicy(actor=actor, critic=critic, model_cfg=model_cfg,
                  actor_lr=1e-4, critic_lr=1e-4, gamma=0.99, tau=0.003, device=device)

    rnd = RNDModule(
        feat_dim=state_dim,
        act_stack_dim=action_stack_dim,
        hidden_dim=512,
        lr=1e-4,
        device=device
    )

    if args.resume_path is not None:
        ARDOR.load(args.resume_path)
        print(f"Loaded model from {args.resume_path}")

    # ----------------- Train buffers -----------------
    replay_buffer = ReplayBuffer(capacity=100000)
    episode_max_steps = env_cfg['episode_max_steps']
    batch_size = 64
    num_episodes = args.num_episodes
    save_interval = args.save_interval

    best_reward = -float('inf')
    best_cr_so_far = -float('inf')
    old_best_path = None
    best_cr_path  = None

    env.reset()

    for ep in range(1, num_episodes + 1):
        # --- Episode-level scalar accumulators (using decomposed rewards from env) ---
        ep_rewards = {
            "total_reward": 0.0,
            "3d_recon": 0.0,
            "collision": 0.0,
            "step_cost": 0.0,
            "occ_ratio": 0.0,
            "curiosity": 0.0,
            "near_target": 0.0,
        }

        traj: list[StepRec] = []
        spikes_cr  = []
        spikes_occ = []
        last_cr    = None
        last_occ_r = None

        # Initial observation (use env internal buffers)
        obs_np = env.obs_buf.clone().cpu().numpy()
        amodal_np = env.amodal_feat_buf.cpu().numpy()

        # --------- Rollout ---------
        for t in range(episode_max_steps):
            s_t  = torch.from_numpy(obs_np).float().to(device)
            sA_t = torch.from_numpy(amodal_np).float().to(device)

            act_t, cand_t, idx_t = ARDOR.select_action(sA_t, s_t, env, env_id=0)

            (next_obs, next_amodal, reward, done,
             tsdf_vol, done_cr, curr_CR, reward_components,
             final_action, amodal_infer_ms) = env.step(act_t.to(device))

            last_done_cr = bool(done_cr[0].item())

            # Intrinsic reward (RND)
            with torch.no_grad():
                r_intr_tensor = rnd.compute_intrinsic_reward(next_amodal, next_obs)
            r_intr = float(reward_cfg['intrinsic_reward_weight'] * r_intr_tensor.cpu().numpy()[0])

            # Extrinsic decomposition from env
            r3d  = float(reward_components["3d_recon"])
            r_occ = float(reward_components["occ_ratio"])
            r_extr_total = float(reward_components["total_reward"])
            # r_base: all other components + intrinsic (so that r_base + r3d + r_occ = total + intrinsic)
            r_base = (r_extr_total - r3d - r_occ) + r_intr

            # cand2 (candidate set at next state s2)
            with torch.no_grad():
                cand2_np = env.sample_candidates_xy(
                    K=model_cfg['candidate_K'], P=model_cfg.get('pitch_P', 5), env_id=0
                ).astype(np.float32)

            # Store trajectory step (push to replay at episode end)
            traj.append(StepRec(
                s=obs_np[0], sA=amodal_np[0], a=final_action.cpu().numpy()[0],
                r_base=r_base, r3d=r3d, r_occ=r_occ,
                s2=next_obs.cpu().numpy()[0], sA2=next_amodal.cpu().numpy()[0],
                d=float(done[0].cpu().numpy()),
                cand=cand_t[0].cpu().numpy(), idx=int(idx_t.item()),
                cand2=cand2_np, t=t
            ))

            # Accumulate for logging
            ep_rewards["total_reward"]   += r_extr_total + r_intr
            ep_rewards["3d_recon"]       += r3d
            ep_rewards["collision"]      += reward_components["collision"]
            ep_rewards["step_cost"]      += reward_components["step_cost"]
            ep_rewards["occ_ratio"]      += reward_components["occ_ratio"]
            ep_rewards["curiosity"]      += r_intr
            ep_rewards["near_target"]    += reward_components["near_target"]

            # CR spike detection
            if last_cr is None:
                last_cr = float(curr_CR.item())
            delta_cr = float(curr_CR.item()) - last_cr
            if (delta_cr >= SPIKE_DELTA_CR) or last_done_cr:
                spikes_cr.append(t)
                last_cr = float(curr_CR.item())

            # OCC spike detection based on the OCC reward component (positive jumps)
            if last_occ_r is None:
                last_occ_r = r_occ
            delta_occ = r_occ - last_occ_r
            if delta_occ >= SPIKE_DELTA_OCC:
                spikes_occ.append(t)
            last_occ_r = r_occ

            # Move to next state
            obs_np = next_obs.cpu().numpy()
            amodal_np = next_amodal.cpu().numpy()

            if bool(done[0].item()):
                break
        
        # ===== Reward redistribution part =====
        T = len(traj)
        final_CR = float(curr_CR.item()) if T > 0 else 0.0
        if T == 0:
            continue

        # -------- Hampel-correct OCC reward series --------
        occ_series_raw = [tr.r_occ for tr in traj]
        occ_series_filt, occ_corrected_idx = _hampel_filter_preserve_steps(
            occ_series_raw, window_size=HAMPEL_WINDOW, n_sigma=HAMPEL_N_SIGMA, hold_len=HAMPEL_HOLD_LEN
        )
        # Apply corrected OCC rewards back into the trajectory
        for i in range(T):
            traj[i].r_occ = float(occ_series_filt[i])
        # Suppress fake spikes near correction indices, then re-interpret spikes
        spikes_occ = _suppress_spikes_near(spikes_occ, occ_corrected_idx, radius=SUPPRESS_RADIUS)
        #------------------------------------------------

        if (len(spikes_cr) == 0) and (len(spikes_occ) == 0):
            # If there are no spikes at all, just push original decomposed rewards
            for tr in traj:
                r_final = tr.r_base + tr.r3d + tr.r_occ
                replay_buffer.push(tr.s, tr.sA, tr.a, r_final,
                                   tr.s2, tr.sA2, tr.d, cand=tr.cand, idx=tr.idx, cand2=tr.cand2)
        else:
            # 1) Initialize bonus arrays for each component
            bonus_3d  = np.zeros(T, dtype=np.float32)
            bonus_occ = np.zeros(T, dtype=np.float32)

            # 2) Redistribute CR component (if any spikes)
            if len(spikes_cr) > 0:
                t_star_cr = spikes_cr[-1]
                R3D_prefix = float(sum(tr.r3d for tr in traj[:t_star_cr+1]))
                steps_cr = list(range(0, t_star_cr+1))
                w_cr = make_weights_multi(
                    spikes=spikes_cr, steps=steps_cr, mode=WEIGHT_MODE,
                    near_lambda=NEAR_LAMBDA_CR, poly_p=POLY_P, mix_eta=MIX_ETA, eta_multi=MULTI_SPIKE_ETA
                )
                bonus = R3D_prefix * w_cr  # redistributed mass over prefix
                bonus_3d[:t_star_cr+1] = bonus  # assign to prefix steps

            # 3) Redistribute OCC component (if any spikes)
            if len(spikes_occ) > 0:
                t_star_occ = spikes_occ[-1]
                ROCC_prefix = float(sum(tr.r_occ for tr in traj[:t_star_occ+1]))
                steps_occ = list(range(0, t_star_occ+1))
                w_occ = make_weights_multi(
                    spikes=spikes_occ, steps=steps_occ, mode=WEIGHT_MODE,
                    near_lambda=NEAR_LAMBDA_OCC, poly_p=POLY_P, mix_eta=MIX_ETA, eta_multi=MULTI_SPIKE_ETA
                )
                bonus = ROCC_prefix * w_occ
                bonus_occ[:t_star_occ+1] = bonus

            # 4) Final push: preserve total mass of each component
            for i in range(T):
                tr = traj[i]
                # On prefix steps, replace original r3d/r_occ with redistributed bonus;
                # after the last spike, keep remaining original tail mass.
                add_r3d  = tr.r3d  if (len(spikes_cr)==0  or i >  t_star_cr)  else 0.0
                add_rocc = tr.r_occ if (len(spikes_occ)==0 or i >  t_star_occ) else 0.0
                r_final = tr.r_base + bonus_3d[i] + bonus_occ[i] + add_r3d + add_rocc
                replay_buffer.push(tr.s, tr.sA, tr.a, r_final,
                                   tr.s2, tr.sA2, tr.d, cand=tr.cand, idx=tr.idx, cand2=tr.cand2)
        
        # Freeze / refresh sampling distribution and log its statistics
        replay_buffer.freeze_distribution(ARDOR.replay_sampling)
        analyze_sampling_distribution(replay_buffer, ARDOR.replay_sampling, writer=writer, step=ep, log_dir=log_dir, tag="Replay")  

        # Reward without curiosity (for checkpoint selection)
        ep_reward_no_curi = ep_rewards["total_reward"] - ep_rewards["curiosity"]

        # ----------------- Checkpoint saving -----------------
        if ep_reward_no_curi > best_reward:
            best_reward = ep_reward_no_curi
            if old_best_path is not None and os.path.exists(old_best_path):
                os.remove(old_best_path)
            new_best_path = os.path.join(log_dir, f"Best_reward={round(best_reward,2)}_epi={ep}.pth")
            ARDOR.save(new_best_path)
            old_best_path = new_best_path
            print(f"  [*] New best reward(no_curi) = {best_reward:.2f} -> {new_best_path}")

        if final_CR > best_cr_so_far:
            best_cr_so_far = final_CR
            if best_cr_path is not None and os.path.exists(best_cr_path):
                os.remove(best_cr_path)
            best_cr_path = os.path.join(log_dir, f"BestCR_model_ep{ep}_CR={best_cr_so_far:.2f}.pth")
            ARDOR.save(best_cr_path)
            print(f"  [*] New best CR = {best_cr_so_far:.2f} -> {best_cr_path}")

        if last_done_cr:
            cr_path = os.path.join(log_dir, f"CR_reached_model_ep{ep}_len={T}.pth")
            ARDOR.save(cr_path)
            print(f"  [*] CR-threshold reached -> {cr_path}")

        # Periodic snapshot
        if ep % save_interval == 0:
            filename = os.path.join(log_dir, f"model_ep{ep}.pth")
            ARDOR.save(filename)
            print(f"  [*] Model saved: {filename}")

        # --------- Offline-style multiple updates per episode ---------
        avg_c_loss = avg_a_loss = avg_rnd_loss = 0.0
        diag_sums = {"Q_mean":0.0,"y_mean":0.0,"y_std":0.0,"p_star_entropy":0.0,"actor_entropy":0.0}
        upd_count = 0
        if len(replay_buffer) >= max(MIN_BUFFER, batch_size):
            update_steps = int(UPDATE_RATIO * max(1, T))
            for _ in range(update_steps):
                c_loss, a_loss, diag = ARDOR.update(replay_buffer, batch_size=batch_size)
                if len(replay_buffer) > batch_size:
                    (s_b, sA_b, a_b, r_b, s2_b, sA2_b, d_b, cand_b, idx_b, cand2_b) = replay_buffer.sample(
                        batch_size, device=device, sampler_cfg=ARDOR.replay_sampling
                    )
                    # Off-policy RND update on next states
                    rnd_loss = rnd.update(sA2_b, s2_b)
                else:
                    rnd_loss = 0.0
                avg_c_loss += float(c_loss)
                avg_a_loss += float(a_loss)
                avg_rnd_loss += float(rnd_loss)
                for k in diag_sums:
                    diag_sums[k] += float(diag.get(k, 0.0))
                upd_count += 1

        if upd_count > 0:
            avg_c_loss /= upd_count
            avg_a_loss /= upd_count
            avg_rnd_loss /= upd_count
            diag_avg = {k: (diag_sums[k] / upd_count) for k in diag_sums}
        else:
            diag_avg = {k: 0.0 for k in diag_sums}
            
        for k,v in diag_avg.items():
            if v != v:  # NaN check
                diag_avg[k] = 0.0

        # ----------------- TensorBoard Logs -----------------
        ep_idx = ep
        writer.add_scalar("Reward/Total_Reward", float(ep_rewards["total_reward"]), ep_idx)
        writer.add_scalar("Reward/3d_recon_Episode", float(ep_rewards["3d_recon"]), ep_idx)
        writer.add_scalar("Reward/Collision_Episode", float(ep_rewards["collision"]), ep_idx)
        writer.add_scalar("Reward/StepCost_Episode", float(ep_rewards["step_cost"]), ep_idx)
        writer.add_scalar("Reward/OccRatio_Episode", float(ep_rewards["occ_ratio"]), ep_idx)
        writer.add_scalar("Reward/Curiosity_Episode", float(ep_rewards["curiosity"]), ep_idx)
        writer.add_scalar("Reward/Near_target_Episode", float(ep_rewards["near_target"]), ep_idx)

        writer.add_scalar("Train/CriticLoss_avg", avg_c_loss, ep_idx)
        writer.add_scalar("Train/ActorLoss_avg", avg_a_loss, ep_idx)
        writer.add_scalar("Train/RNDLoss_avg", avg_rnd_loss, ep_idx)
        
        writer.add_scalar("Diag/Q/mean",         diag_avg["Q_mean"], ep_idx)
        writer.add_scalar("Diag/TargetY/mean",   diag_avg["y_mean"], ep_idx)
        writer.add_scalar("Diag/TargetY/std",    diag_avg["y_std"], ep_idx)
        writer.add_scalar("Diag/p_star/entropy", diag_avg["p_star_entropy"], ep_idx)
        writer.add_scalar("Diag/actor/entropy",  diag_avg["actor_entropy"], ep_idx)

        print(f"[end-prefix] Ep {ep}/{num_episodes} | len={T} "
              f"spikes_cr={spikes_cr} spikes_occ={spikes_occ} "
              f"| CR={final_CR:.3f} | buf={len(replay_buffer)} "
              f"| losses(c,a,rnd)=({avg_c_loss:.4f},{avg_a_loss:.4f},{avg_rnd_loss:.4f})")
     
    # Save final model
    ARDOR.save(os.path.join(log_dir, "model_final.pth"))
    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    main()
