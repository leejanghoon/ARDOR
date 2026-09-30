# ARDOR_test.py
import argparse
import os
import re
import pickle
import torch
import numpy as np
import cv2
import genesis as gs
import json
import time
import sys
from pathlib import Path

from _3D_recon.utils import (
    mesh_from_tsdf_safe_colored,
    precision_recall_fscore_chamfer
)
from model.ARDOR_env import ARDOR_Env
from model.ARDOR_policy import Actor, Critic, ARDORPolicy

# ---------- utils ----------
def to_jsonable(x):
    import numpy as _np
    import torch as _torch
    if isinstance(x, _torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return x

def parse_scenes_arg(s):
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    s = str(s).replace(",", " ")
    return [int(tok) for tok in s.split() if tok.strip()]

def safe_avg(x):
    return float(np.mean(x)) if len(x) > 0 else None

def find_latest_cr_ckpt(log_dir):
    """
    Priority for selecting the latest checkpoint:
      1) CR_reached_model_ep{N}_len=*.pth with largest ep (tie-breaker: latest mtime)
      2) BestCR_model_ep*.pth (latest mtime)
      3) Best_reward=*.pth (latest mtime)
      4) model_final.pth
      5) model_ep*.pth (latest mtime)
    """
    if not os.path.isdir(log_dir):
        return None
    files = os.listdir(log_dir)

    # 1) CR_reached
    cr = []
    pat = re.compile(r"^CR_reached_model_ep(\d+)_len=(\d+)\.pth$")
    for f in files:
        m = pat.match(f)
        if m:
            ep = int(m.group(1))
            p = os.path.join(log_dir, f)
            cr.append((ep, os.path.getmtime(p), p))
    if cr:
        cr.sort(key=lambda x: (x[0], x[1]))
        return cr[-1][2]

    # 2) BestCR
    bestcr = [os.path.join(log_dir, f) for f in files
              if f.startswith("BestCR_model_ep") and f.endswith(".pth")]
    if bestcr:
        bestcr.sort(key=lambda p: os.path.getmtime(p))
        return bestcr[-1]

    # 3) Best_reward
    bestr = [os.path.join(log_dir, f) for f in files
             if f.startswith("Best_reward=") and f.endswith(".pth")]
    if bestr:
        bestr.sort(key=lambda p: os.path.getmtime(p))
        return bestr[-1]

    # 4) model_final
    mf = os.path.join(log_dir, "model_final.pth")
    if os.path.exists(mf):
        return mf

    # 5) any model_ep*.pth
    anyep = [os.path.join(log_dir, f) for f in files
             if f.startswith("model_ep") and f.endswith(".pth")]
    if anyep:
        anyep.sort(key=lambda p: os.path.getmtime(p))
        return anyep[-1]

    return None

@torch.no_grad()
def select_action_eval_Q(ARDOR, amodal_flat, action_stack, env):
    """Select action by Q-max over candidates + log Actor outputs for debugging."""
    cand = ARDOR._sample_candidates(env, env_id=0)          # (1, K, 3)
    B, K, _ = cand.shape
    c_flat  = cand.view(B * K, 3)
    sA_rep  = amodal_flat.repeat_interleave(K, dim=0)
    s_rep   = action_stack.repeat_interleave(K, dim=0)
    q_all   = ARDOR.critic(sA_rep, s_rep, c_flat).view(B, K)
    idx = int(torch.argmax(q_all, dim=1).item())
    act = cand[:, idx, :]
    idx_t = torch.tensor([idx], dtype=torch.long, device=ARDOR.device)

    # Actor logits/probs (for debugging)
    logits = ARDOR.actor(amodal_flat, action_stack, cand)    # (1, K)
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    dbg = {
        "logits": logits,
        "probs": probs,
        "entropy": entropy,
        "epsilon": torch.tensor(
            [getattr(ARDOR, "eps_snapshot", getattr(ARDOR, "eps", 0.0))],
            device=logits.device
        ),
        "explore": torch.tensor([0], device=logits.device),
    }
    return act, cand, idx_t, dbg

def get_current_house_scene(env):
    epi_per = env.env_cfg.get("epi_per_scene_house", 1)
    gi = (env._epi_count - 1) // epi_per
    n_scenes = len(env.scene_list)
    scene_idx = gi % n_scenes
    house_idx = (gi // n_scenes) % len(env.house_list)
    return env.house_list[house_idx], env.scene_list[scene_idx]

# ---------- core: evaluate a single scene with the latest checkpoint ----------
def evaluate_scene(args, scene):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # per-scene log dir & cfg
    log_dir = os.path.join("logs", f"{args.exp_name}_{scene}")
    cfg_path = os.path.join(log_dir, "cfg.pkl")
    assert os.path.exists(cfg_path), f"[scene {scene}] cfg.pkl not found: {cfg_path}"
    env_cfg, obs_cfg, reward_cfg, model_cfg, amodal_cfg = pickle.load(open(cfg_path, "rb"))

    # test-time episodes per house
    env_cfg["epi_per_scene_house"] = int(args.episodes_per_house)

    # output dir (separated per scene)
    save_root = os.path.join(args.save_dir, f"{args.exp_name}_scene{scene}")
    os.makedirs(save_root, exist_ok=True)

    # env
    env = ARDOR_Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        amodal_cfg=amodal_cfg,
        model_cfg=model_cfg,
        show_viewer=args.viewer,
        render=args.render,
        device=device,
        inference=True
    )
    # Fix scene and set depth noise according to flag
    env.scene_list = [scene]
    env.depth_noise = bool(args.depth_noise)

    # model
    state_dim = model_cfg['state_dim']
    action_dim = model_cfg["action_dim"]
    action_stack_dim = model_cfg['action_stack_dim']
    actor  = Actor(state_dim, action_stack_dim, action_dim)
    critic = Critic(state_dim, action_stack_dim, action_dim)
    ARDOR = ARDORPolicy(actor=actor, critic=critic, model_cfg=model_cfg, device=device)

    # choose checkpoint
    ckpt_path = find_latest_cr_ckpt(log_dir)
    assert ckpt_path is not None and os.path.exists(ckpt_path), f"[scene {scene}] checkpoint not found in {log_dir}"
    ARDOR.load(ckpt_path)
    print(f"[scene {scene}] Loaded ARDOR from {ckpt_path}")

    # deterministic evaluation
    ARDOR.eps = 0.0
    ARDOR.eps_end = 0.0

    # total episodes
    num_eps = len(env.house_list) * int(args.episodes_per_house)

    all_episode_data = []
    house_summaries = {h: {"rec": [], "cd": [], "f1": [], "prec": [], "reach_steps": [], "reached": 0}
                       for h in env.house_list}

    env.reset()
    for epi_idx in range(num_eps):
        house_num, scene_num = get_current_house_scene(env)
        print(f"[scene {scene}] ======== Episode {epi_idx} | House {house_num} | Scene {scene_num} ========")

        ep_dir = os.path.join(save_root, f"ep_{epi_idx:03d}_H{int(house_num)}_S{int(scene_num)}")
        os.makedirs(ep_dir, exist_ok=True)
        img_dir = os.path.join(ep_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        step_i = 0
        init_rec = float(getattr(env, "prev_metric", torch.tensor(0.0)).item())
        cam_pos0 = list(map(float, env.camera.pos))
        cam_look0 = list(map(float, env.camera.lookat))

        episode_step_data = [{
            "step": 0,
            "pos": cam_pos0,
            "lookat": cam_look0,
            "recall": init_rec,
            "precision": None,
            "f1": None,
            "cd": None
        }]

        # Step 0 images (reset-safe)
        try:
            if getattr(env, "last_rgb", None) is not None:
                cv2.imwrite(
                    os.path.join(img_dir, f"step_{step_i:03d}_rgb.png"),
                    cv2.cvtColor(env.last_rgb, cv2.COLOR_BGR2RGB)
                )
            if getattr(env, "last_amodal_mask", None) is not None:
                am0 = env.last_amodal_mask
                if am0.dtype != np.uint8:
                    am0 = am0.astype(np.uint8)
                cv2.imwrite(os.path.join(img_dir, f"step_{step_i:03d}_amodal.png"), (am0 * 255))
            if getattr(env, "last_occluded_mask", None) is not None:
                oc0 = env.last_occluded_mask
                if oc0.dtype != np.uint8:
                    oc0 = oc0.astype(np.uint8)
                cv2.imwrite(os.path.join(img_dir, f"step_{step_i:03d}_occluded.png"), (oc0 * 255))
        except Exception:
            pass

        # Detailed per-step log for JSON output
        steps_log = [{
            "step": 0,
            "pos": cam_pos0,
            "lookat": cam_look0,
            "recall": init_rec,
            "precision": None,
            "f1": None,
            "cd": None,
            "policy": None,
            "rewards": None,
            "occ_ratio_val": float(getattr(env, "curr_occ_ratio", 0.0)),
        }]

        reached = False
        reach_step = None

        while True:
            step_i += 1

            # --- PRE-STEP SNAPSHOT (reset-safe for metrics & pose) ---
            pre_epi_count = getattr(env, "_epi_count", None)
            mesh_gt_this_step = env.mesh_gt
            voxel_size_this_step = getattr(env, "voxel_size", None)
            pre_cam_pos  = list(map(float, env.camera.pos))
            pre_cam_look = list(map(float, env.camera.lookat))
            # ---------------------------------------------------------

            # obs/amodal → tensor
            obs_np = env.obs_buf.clone().cpu().numpy()
            amodal_np = env.amodal_feat_buf.cpu().numpy()
            action_stack = torch.from_numpy(obs_np).float().to(device)
            amodal_flat  = torch.from_numpy(amodal_np).float().to(device)

            # Action selection by Q-max
            act, cand, idx_t, dbg = select_action_eval_Q(ARDOR, amodal_flat, action_stack, env)

            # Candidate metadata (grouping into (dx,dy)+pitch sets, chosen info, q_chosen)
            cand_groups = None
            chosen_abs = None
            g_idx = None
            p_idx = None
            q_chosen = None

            if cand is not None and idx_t is not None and dbg is not None:
                with torch.no_grad():
                    B, K, _ = cand.shape
                    c_flat = cand.view(B * K, 3)
                    sA_rep = amodal_flat.repeat_interleave(K, dim=0)
                    s_rep  = action_stack.repeat_interleave(K, dim=0)
                    qk = ARDOR.critic(sA_rep, s_rep, c_flat).view(B, K)

                    P_used = int(ARDOR.model_cfg.get('pitch_P', 5))
                    cand_np = cand[0].detach().cpu().numpy()
                    C = cand_np.shape[0]; assert C % P_used == 0
                    G = C // P_used
                    cand_g = cand_np.reshape(G, P_used, 3)
                    dxdy_g = cand_g[:, 0, :2]
                    dp_rad_g = cand_g[:, :, 2]
                    dp_deg_g = np.degrees(dp_rad_g)
                    pos0 = env.pos_xy[0].detach().cpu().numpy()
                    pitch0_deg = float(env.pitch[0].item())
                    x_abs_g = pos0[0] + dxdy_g[:, 0]
                    y_abs_g = pos0[1] + dxdy_g[:, 1]
                    pitch_abs_deg_g = np.clip(
                        pitch0_deg + dp_deg_g,
                        env.pitch_range[0],
                        env.pitch_range[1]
                    )

                    logits_np = dbg["logits"][0].detach().cpu().numpy()
                    probs_np  = dbg["probs"][0].detach().cpu().numpy()
                    qk_np     = qk[0].detach().cpu().numpy()
                    logits_g = logits_np.reshape(G, P_used)
                    probs_g  = probs_np.reshape(G, P_used)
                    qk_g     = qk_np.reshape(G, P_used)
                    idx_flat = int(idx_t[0].item())
                    g_idx = idx_flat // P_used
                    p_idx = idx_flat % P_used
                    chosen_abs = [
                        float(x_abs_g[g_idx]),
                        float(y_abs_g[g_idx]),
                        float(pitch_abs_deg_g[g_idx, p_idx])
                    ]
                    q_chosen = float(qk_g[g_idx, p_idx])

                    # cand_groups for JSON
                    cand_groups = []
                    for g in range(G):
                        cand_groups.append({
                            "xy_delta": [ float(dxdy_g[g,0]), float(dxdy_g[g,1]) ],
                            "xy_abs":   [ float(x_abs_g[g]),  float(y_abs_g[g])  ],
                            "pitch_rad":  [ float(v) for v in dp_rad_g[g].tolist() ],
                            "pitch_deg":  [ float(v) for v in pitch_abs_deg_g[g].tolist() ],
                            "logits":     [ float(v) for v in logits_g[g].tolist() ],
                            "probs":      [ float(v) for v in probs_g[g].tolist() ],
                            "q_values":   [ float(v) for v in qk_g[g].tolist() ],
                        })

            # Take environment step
            (next_obs, next_amodal_feat, reward, done,
             tsdf_vol, done_cr, curr_recall, reward_components,
             final_action, amodal_infer_ms) = env.step(act.to(device))

            # Detect automatic reset (house/episode boundary)
            post_epi_count = getattr(env, "_epi_count", None)
            auto_reset = (
                pre_epi_count is not None and
                post_epi_count is not None and
                post_epi_count != pre_epi_count
            )

            # Recall is logged using the current_metric snapshot before any reset
            step_rec_for_log = float(
                reward_components.get(
                    "current_metric",
                    float(curr_recall.item())
                )
            )

            print(
                f"[scene {scene}] Step {step_i} | reward: {reward.item():.4f} "
                f"| Recall(current_metric): {step_rec_for_log:.4f}"
                + (" | AUTO-RESET" if auto_reset else "")
            )

            if bool(done_cr[0].item()) and (not reached):
                reached = True
                reach_step = step_i

            # TD target / error (for debugging)
            y_target = None
            td_error = None
            try:
                if q_chosen is None:
                    raise ValueError("q_chosen is None")
                if not bool(done[0].item()):
                    cand2 = ARDOR._sample_candidates(env, env_id=0)
                    B2, K2, _ = cand2.shape
                    c2_flat = cand2.view(B2 * K2, 3)
                    sA2_rep = torch.from_numpy(next_amodal_feat).float().to(device).repeat_interleave(K2, dim=0)
                    s2_rep  = torch.from_numpy(next_obs).float().to(device).repeat_interleave(K2, dim=0)
                    with torch.no_grad():
                        q_next_online = ARDOR.critic(sA2_rep, s2_rep, c2_flat).view(B2, K2)
                        a_star = q_next_online.argmax(dim=1, keepdim=True)
                        q_next_targ = ARDOR.target_critic(sA2_rep, s2_rep, c2_flat).view(B2, K2)
                        Q_next = q_next_targ.gather(1, a_star).item()
                    y_target = float(reward.item() + ARDOR.gamma * Q_next)
                else:
                    y_target = float(reward.item())
                td_error = float(y_target - q_chosen)
            except Exception:
                pass

            # Mesh-based evaluation at fixed interval / termination
            # (must use the pre-step GT mesh and voxel_size snapshot)
            if (step_i % args.eval_every == 0) or bool(done[0].item()):
                mesh_rec = mesh_from_tsdf_safe_colored(tsdf_vol, min_weight=0.5)
                if (mesh_rec is None) or (mesh_rec.faces.size == 0):
                    step_prec = None
                    step_f1   = None
                    step_cd   = None
                else:
                    tau_voxel = (
                        voxel_size_this_step
                        if voxel_size_this_step is not None
                        else getattr(env, "voxel_size", None)
                    )
                    step_prec, _rec_ign, step_f1, step_cd = precision_recall_fscore_chamfer(
                        mesh_gt_this_step, mesh_rec, tau=tau_voxel, n_samples=None
                    )
            else:
                step_prec = None
                step_f1   = None
                step_cd   = None

            # Camera info: if reset happened or missing, fall back to pre-step snapshot
            cam_pos   = to_jsonable(reward_components.get("camera_pos"))
            cam_look  = to_jsonable(reward_components.get("camera_lookat"))
            if cam_pos is None or auto_reset:
                cam_pos  = pre_cam_pos
            if cam_look is None or auto_reset:
                cam_look = pre_cam_look
            applied_act = reward_components.get(
                "applied_action",
                to_jsonable(act[0].detach().cpu().numpy())
            )

            # Detailed per-step log (JSON-friendly schema)
            step_payload = {
                "step": step_i,
                "pos": cam_pos,
                "lookat": cam_look,
                "recall": step_rec_for_log,
                "precision": step_prec,
                "f1": step_f1,
                "cd": step_cd,
                "policy": {
                    "cand_groups": cand_groups,
                    "chosen": {
                        "group_idx": (int(g_idx) if g_idx is not None else None),
                        "pitch_idx": (int(p_idx) if p_idx is not None else None),
                        "flat_idx":  (int(idx_t[0].item()) if idx_t is not None else None),
                        "delta":     to_jsonable(applied_act),
                        "pitch_deg": (
                            float(pitch_abs_deg_g[g_idx, p_idx])
                            if (g_idx is not None and p_idx is not None and 'pitch_abs_deg_g' in locals())
                            else None
                        ),
                        "abs":       (to_jsonable(chosen_abs) if chosen_abs is not None else None),
                        "q_chosen":  (float(q_chosen) if q_chosen is not None else None),
                        "y_target":  (float(y_target) if y_target is not None else None),
                        "td_error":  (float(td_error) if td_error is not None else None)
                    },
                },
                "rewards": {
                    "total": float(reward_components["total_reward"]),
                    "recon_3d": float(reward_components["3d_recon"]),
                    "step_cost": float(reward_components["step_cost"]),
                    "occ_ratio": float(reward_components["occ_ratio"]),
                    "near_target": float(reward_components["near_target"]),
                },
                "occ_ratio_val": float(reward_components.get("occ_ratio_val", 0.0)),
            }
            steps_log.append(step_payload)

            # Lightweight summary for legacy code compatibility
            episode_step_data.append({
                "step": step_i,
                "pos": cam_pos,
                "lookat": cam_look,
                "recall": step_rec_for_log,
                "precision": step_prec,
                "f1": step_f1,
                "cd": step_cd
            })

            # Save images for this step (reset-safe)
            try:
                frame_bgr = reward_components.get("frame_bgr", None)
                am_mask   = reward_components.get("amodal_mask", None)
                oc_mask   = reward_components.get("occluded_mask", None)
                if frame_bgr is not None:
                    cv2.imwrite(
                        os.path.join(img_dir, f"step_{step_i:03d}_rgb.png"),
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    )
                if am_mask is not None:
                    am = am_mask.astype(np.uint8) if am_mask.dtype != np.uint8 else am_mask
                    cv2.imwrite(os.path.join(img_dir, f"step_{step_i:03d}_amodal.png"), (am * 255))
                if oc_mask is not None:
                    oc = oc_mask.astype(np.uint8) if oc_mask.dtype != np.uint8 else oc_mask
                    cv2.imwrite(os.path.join(img_dir, f"step_{step_i:03d}_occluded.png"), (oc * 255))
            except Exception:
                pass

            # Termination conditions
            if bool(done[0].item()) or (step_i >= args.max_steps):
                final_rec  = episode_step_data[-1]["recall"]
                final_cd   = episode_step_data[-1]["cd"]
                final_f1   = episode_step_data[-1]["f1"]
                final_prec = episode_step_data[-1]["precision"]

                if final_rec  is not None:
                    house_summaries[house_num]["rec"].append(final_rec)
                if final_cd   is not None:
                    house_summaries[house_num]["cd"].append(final_cd)
                if final_f1   is not None:
                    house_summaries[house_num]["f1"].append(final_f1)
                if final_prec is not None:
                    house_summaries[house_num]["prec"].append(final_prec)
                if reached:
                    house_summaries[house_num]["reached"] += 1
                    house_summaries[house_num]["reach_steps"].append(reach_step)

                # Total camera path length in 3D (sum of step-to-step distances)
                episode_distance = 0.0
                for i in range(1, len(episode_step_data)):
                    p0 = np.array(episode_step_data[i-1]["pos"])
                    p1 = np.array(episode_step_data[i]["pos"])
                    episode_distance += float(np.linalg.norm(p1 - p0))

                all_episode_data.append({
                    "infer_idx": epi_idx,
                    "house": int(house_num),
                    "scene": int(scene_num),
                    "reached": bool(reached),
                    "reach_step": (int(reach_step) if reach_step is not None else None),
                    "recall": (float(final_rec) if final_rec is not None else None),
                    "precision": (float(final_prec) if final_prec is not None else None),
                    "f1": (float(final_f1) if final_f1 is not None else None),
                    "cd": (float(final_cd) if final_cd is not None else None),
                    "steps": int(step_i),
                    "dist": float(episode_distance),
                    "cam_trajectory": episode_step_data,
                })
                # Per-episode JSON (same schema as step logs)
                episode_json = {
                    "episode_idx": int(epi_idx),
                    "house": int(house_num),
                    "scene": int(scene_num),
                    "summary": {
                        "reached": bool(reached),
                        "reach_step": (int(reach_step) if reach_step is not None else None),
                        "final": {
                            "recall": (float(final_rec) if final_rec is not None else None),
                            "precision": (float(final_prec) if final_prec is not None else None),
                            "f1": (float(final_f1) if final_f1 is not None else None),
                            "cd": (float(final_cd) if final_cd is not None else None),
                        },
                        "steps": int(step_i),
                        "distance": float(episode_distance),
                    },
                    "steps": steps_log,
                }
                with open(os.path.join(ep_dir, "episode.json"), "w") as jf:
                    json.dump(episode_json, jf, indent=2)

                break

    # ---- Per-house summary (same JSON-ish schema as training side) ----
    house_summary_rows = []
    for h in env.house_list:
        recs  = house_summaries[h]["rec"]
        cds   = house_summaries[h]["cd"]
        f1s   = house_summaries[h]["f1"]
        pres  = house_summaries[h]["prec"]
        rsteps= house_summaries[h]["reach_steps"]
        row = {
            "house": int(h),
            "avg_recall": safe_avg(recs),
            "avg_cd": safe_avg(cds),
            "avg_f1": safe_avg(f1s),
            "avg_precision": safe_avg(pres),
            "reach_rate": house_summaries[h]["reached"] / float(args.episodes_per_house),
            "avg_reach_step": safe_avg(rsteps)
        }
        house_summary_rows.append(row)

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_all_path = os.path.join(save_root, f"{args.exp_name}_S{scene}_episodes_{ts}.json")
    with open(json_all_path, "w") as f:
        json.dump(all_episode_data, f, indent=2)
    print(f"[scene {scene}] Saved all-episodes JSON: {json_all_path}")

    json_summary_path = os.path.join(save_root, f"{args.exp_name}_S{scene}_house_summary_{ts}.json")
    with open(json_summary_path, "w") as f:
        json.dump(house_summary_rows, f, indent=2)
    print(f"[scene {scene}] Saved per-house summary JSON: {json_summary_path}")

    # Clean up resources
    del ARDOR, actor, critic, env
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return a compact summary for this scene
    all_rec = [e["recall"] for e in all_episode_data if e["recall"] is not None]
    all_cd  = [e["cd"] for e in all_episode_data if e["cd"] is not None]
    all_f1  = [e["f1"] for e in all_episode_data if e["f1"] is not None]
    all_pr  = [e["precision"] for e in all_episode_data if e["precision"] is not None]
    return {
        "scene": scene,
        "overall": {
            "avg_recall": safe_avg(all_rec),
            "avg_cd": safe_avg(all_cd),
            "avg_f1": safe_avg(all_f1),
            "avg_precision": safe_avg(all_pr),
        },
        "save_root": save_root,
        "ckpt": ckpt_path
    }

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument(
        "--scenes",
        nargs="*",
        default='',
        help="e.g., 3 4 7 or '3,4,7'"
    )
    parser.add_argument("--save_dir", type=str, default="./qualitative_results")
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--episodes_per_house", type=int, default=3)
    # depth noise option (toggle for evaluation)
    parser.add_argument(
        "--depth_noise",
        action="store_true",
        default=False,
        help="Enable depth noise during evaluation"
    )
    args = parser.parse_args()
    gs.init(logging_level="warning")

    # Parse scenes argument
    if args.scenes is None or len(args.scenes) == 0:
        raise ValueError("Please specify --scenes (e.g., --scenes 3 4 7)")
    scenes = parse_scenes_arg(args.scenes)

    all_scene_results = []
    for sc in scenes:
        print(f"\n==================== EVALUATE SCENE {sc} ====================\n")
        res = evaluate_scene(args, sc)
        all_scene_results.append(res)

    print("\n==================== SUMMARY (ALL SCENES) ====================")
    for r in all_scene_results:
        s = r["scene"]; ov = r["overall"]
        print(
            f"Scene {s} | Recall: {ov['avg_recall']} | CD: {ov['avg_cd']} "
            f"| F1: {ov['avg_f1']} | Prec: {ov['avg_precision']}"
        )
        print(f"  ckpt: {r['ckpt']}")
        print(f"  out : {r['save_root']}")

if __name__ == "__main__":
    main()
