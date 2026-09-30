# ARDOR_env.py
import torch
import math
import genesis as gs
import numpy as np
import cv2
import sys
from PIL import Image
from scene_utils import train_scene_build, test_scene_build_omniobject3d, test_scene_build_house3k, line_segment_colliding_with_obstacles_2d_obb,load_obstacles_2d
from _3D_recon.utils import entity_to_trimesh,precision_recall_fscore_chamfer,mesh_from_tsdf_safe_colored
from _3D_recon.fusion import TSDFVolume
sys.path.append('/workspace/Workspace/Genesis/')
from uoais.tools.infer import init_UOIAS, infer_UOIAS_light
import random
import json
import time
import matplotlib.pyplot as plt

def get_azimuth_for_scene(scene_num: int, json_path="/_targets/_scene_init_pos.json") -> float:
    
   
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  

   
    for item in data:
        if item["scene"] == scene_num:
            return item["azimuth"]
    
    
    raise ValueError(f"Scene {scene_num} not found in {json_path}")


class ARDOR_Env:

    def __init__(self,
                 num_envs,
                 env_cfg,
                 obs_cfg,
                 reward_cfg,
                 amodal_cfg,
                 model_cfg,
                 show_viewer=False,
                 render=False,
                 device="cuda",
                 c = 1, # 1 for gray input, 3 for RGB input
                 inference = False
                 ):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.amodal_cfg = amodal_cfg
        self.model_cfg = model_cfg
        self.house_id = 0
        self.Nx = 50 
        self.inference = inference
        self.render = render
        self.show_viewer = show_viewer
        
        self.k = obs_cfg.get("frame_stack", 2) 
        self.dt = env_cfg.get("dt", 0.1)
        self.episode_max_steps = env_cfg.get("episode_max_steps", 50)

        # camera distance/rotation range
        self.pos_xy = torch.zeros((num_envs, 2), device=self.device) 
        self.pitch  = torch.zeros((num_envs,), device=self.device)
        self.distances = torch.ones((num_envs,), device=self.device)  

        
        self.distance_range = env_cfg.get("distance_range", [6, 25])
        self.pitch_range    = env_cfg.get("pitch_range", [0, 50])
        self.pitch_delta_min, self.pitch_delta_max = self.env_cfg["pitch_scale"] 
        
    
        self._3d_recon_weight = self.reward_cfg.get("_3d_recon", 10.0)
        self.cr_threshold = self.reward_cfg.get("CR_threshold", 0.95)
        self.cr_done_reward = self.reward_cfg.get("CR_done_reward", 100.0)
        self.step_cost = self.reward_cfg.get("step_cost", -2.0)
        self.occ_ratio_reward_weight = self.reward_cfg.get("occ_ratio_reward_weight", 0.0)
        self.target_visibility_weight = self.reward_cfg.get("target_visibility", 0.0)
        self.near_target_weight=self.reward_cfg.get("near_target",0.0)
     
        self.c = c  # gray, c = 1
        self.amodal_max_instance = 1
        self.amodal_feature_dim = self.amodal_max_instance*256

        self.amodal_feat_buf = torch.zeros(
            (self.num_envs, self.k, self.amodal_feature_dim),  
            device=self.device
        )
      
        self.obs_buf = torch.zeros(
            (num_envs, self.c*self.k, self.model_cfg['action_dim']), 
            device=self.device
        )

        self.rew_buf = torch.zeros((num_envs,), device=self.device)
        self.reset_buf = torch.ones((num_envs,), device=self.device, dtype=torch.int)
        self.episode_max_steps_buf = torch.zeros((num_envs,), device=self.device, dtype=torch.int)

        

 
        self.prev_metric = torch.zeros((num_envs,), device=self.device)
        self.init_cr_tensor = torch.zeros((self.num_envs,), device=self.device)

        self.last_occ_ratio=1.0
        self.curr_occ_ratio=1.0

        self.obstacles_2d = []
        self._epi_count = 0
        
        # for scene building
        self.scene_list = [env_cfg.get("scene")] 

        # house3k ##################
        # self.house_list = list(i for i in range(1, 51))
        if self.inference:
            self.house_list = [2,4,17,21,25,28,42,43,44,45]
            # self.house_list = [40,26,31,27,34]
        else:
            # self.house_list = [26,27,29,30,31,32,33,34,35,36,37,38,39,40,41,46,47,48,49,50] # scene3 retrain
            self.house_list = [50, 33, 41, 29, 15, 35, 49, 37, 14, 46, 38, 30, 19, 6, 20, 7, 10, 9, 12, 26, 11, 36, 31, 47, 18, 22, 16, 1, 13, 48, 8, 39, 5, 27, 3, 34, 23, 24, 40, 32]

        ############################

        # omniobj3d ###############
        # if self.inference:
        #     house_num_list_raw = list(range(1, 45)) # 1~44 houses
        #     to_remove = {14,15,27,39,43,45} # remove these house numbers due to the error
        #     self.house_list = [x for x in house_num_list_raw if x not in to_remove]
        ###########################

        self.prev_range = None
        self.debug = env_cfg.get("debug", False)
        self.prev_pos_xy = torch.zeros_like(self.pos_xy)  

        self.recon_total_target = self.reward_cfg.get("recon_total_target", 50.0)  
        self.recon_shape = self.reward_cfg.get("recon_shape", "exp")  
        self.recon_gamma = self.reward_cfg.get("recon_gamma", 2.0)
        self.recon_beta  = self.reward_cfg.get("recon_beta", 2.0)     
        self.recon_topoff_on_done = self.reward_cfg.get("recon_topoff_on_done", True)
        self.recon_topoff_tol = self.reward_cfg.get("recon_topoff_tol", 1e-3)
        
        self.depth_noise = env_cfg.get("depth_noise")  
        self.debug_imshow = env_cfg.get("debug_imshow", False)
        print(f"depth_noise: {self.depth_noise}")

    def step(self, actions, debugData=None):
    
        dx = actions[:,0]; dy = actions[:,1]
        dpit_rad = actions[:,2]
      
        dpit_deg = torch.rad2deg(dpit_rad).clamp(self.pitch_delta_min, self.pitch_delta_max)

   
        self.prev_pos_xy = self.pos_xy.clone()
        
      
        self.pos_xy[:,0] += dx
        self.pos_xy[:,1] += dy
        self.pitch += dpit_deg
        self.pitch = self.pitch.clamp(*self.pitch_range)

       
        el = 0.4
        r = torch.linalg.norm(self.pos_xy, dim=1)
        self.distances = (r / math.cos(el)).clamp(*self.distance_range)

   
        self._update_camera_pose()

    
        final_action = torch.stack([dx, dy, torch.deg2rad(dpit_deg)], dim=1)
	self.obs_buf = torch.roll(self.obs_buf, shifts=-1, dims=1)
        self.obs_buf[:, -1, :] = final_action 
	
        if self.debug:
            print("final action (Δd, Δaz(rad), Δpit(rad)):", final_action)

       

     
        _, rgb_img, depth_img, seg_img = self._get_frames()

        
        depth_img_noised = depth_img
        mask = (seg_img == self.house_id)
        depth_img_noised[depth_img_noised >= 40] = 0
        depth_img_noised[~mask] = 0
        intr = self.camera.intrinsics
        extr = np.linalg.inv(self.camera.extrinsics)
        noise_cfg = None
        if self.depth_noise:
            noise_cfg = dict(
                seed=42,
                base_std=1.0/6.0,
                noise_start=5.0,
                skip_ge=40.0,
                debug_imshow=self.debug_imshow,  
            )
        self.tsdf_vol.integrate(rgb_img, depth_img_noised, intr, extr, obs_weight=1.0, noise_cfg=noise_cfg)
            
        if self.inference:
            tsdf_vol_clone = self.tsdf_vol.get_snapshot()
        else:
            tsdf_vol_clone = None

        # Amodal
        rgb_for_amodal = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        with torch.cuda.amp.autocast():
            preds, pred_visibles, amodal_feature, amodal_feature_2, pred_classes, scores = infer_UOIAS_light(
                rgb_for_amodal, depth_img, self.predictor, self.fg_model,
                self.W, self.H, self.cfg, self.args
            )
        amodal_inference_time = (time.time() - t0) * 1000.0

        preds = preds.numpy()
        pred_visibles = pred_visibles.detach().cpu().numpy()
        amodal_feature = amodal_feature_2.detach().cpu().numpy()
        pred_classes = pred_classes.detach().cpu().numpy() if pred_classes is not None else None
        scores = scores.detach().cpu().numpy() if scores is not None else None

        # occlusion ratio
        target_indices = np.where(pred_classes == 0)[0] if pred_classes is not None else []
        if len(target_indices) == 0:
            occ_ratio = 1.0
            target_amodal_mask = np.zeros_like(depth_img, dtype=np.uint8)
            target_visible_mask = np.zeros_like(depth_img, dtype=np.uint8)
            target_occluded_mask = np.zeros_like(depth_img, dtype=np.uint8)
        else:
            best_idx = target_indices[np.argmax(scores[target_indices])]
            target_amodal_mask = preds[best_idx]
            target_visible_mask = pred_visibles[best_idx]
            target_occluded_mask = target_amodal_mask & ~target_visible_mask
            kernel = np.ones((3, 3), np.uint8)
            target_occluded_mask = cv2.morphologyEx(
                target_occluded_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=3
            )
            occ_ratio = np.sum(target_occluded_mask) / (np.sum(target_amodal_mask) + 1e-10)
        self.curr_occ_ratio = float(occ_ratio)
        
     
        self.last_rgb = rgb_img.copy()
        self.last_amodal_mask = target_amodal_mask.astype(np.uint8)
        self.last_visible_mask = target_visible_mask.astype(np.uint8)
        self.last_occluded_mask = target_occluded_mask.astype(np.uint8)

 
        amodal_feature_tensor = torch.tensor(amodal_feature, dtype=torch.float32).to(self.device)
        amodal_feat_vec = self._pool_and_select_topk(amodal_feature_tensor, scores, K=self.amodal_max_instance)
        self.amodal_feat_buf = torch.roll(self.amodal_feat_buf, shifts=-1, dims=1)
        self.amodal_feat_buf[0, -1, :] = amodal_feat_vec
        
        if self.render:
            vis = (target_amodal_mask.astype(np.uint8) * 255) if len(target_indices) > 0 else np.zeros_like(depth_img, dtype=np.uint8)
            cv2.imshow('amodal_mask_target', vis)
            # cv2.waitKey(1)

       
        rew_3d, current_metric_tensor = self._reward_3d_recon()
        current_metric_tensor_clone = current_metric_tensor.clone()
        rew_step_cost = self._reward_step_cost()
        rew_occ_ratio = self._reward_occ_ratio()
        rew_near_target = self._reward_near_target()

      
        self.episode_max_steps_buf += 1
        done_timeup = self.episode_max_steps_buf >= self.episode_max_steps
        mask_cr_done = (current_metric_tensor > self.cr_threshold)
        rew_3d[mask_cr_done] += self.cr_done_reward
        done_cr = mask_cr_done
        done_total = done_timeup | done_cr
        self.reset_buf = done_total.clone()

        if self.debug:
            print(f"Current CR: {current_metric_tensor.item():.4f}")

        self.rew_buf = rew_3d + rew_step_cost + rew_occ_ratio + rew_near_target

        reward_components = {
            "total_reward": float(self.rew_buf.item()),
            "3d_recon": float(rew_3d.item()),
            "collision": 0.0,
            "step_cost": float(rew_step_cost.item()),
            "occ_ratio": float(rew_occ_ratio),
            "near_target": float(rew_near_target.item()),

            # ---- reset-safe snapshots ----
            "camera_pos": np.array(self.camera.pos, dtype=np.float32).copy(),
            "camera_lookat": np.array(self.camera.lookat, dtype=np.float32).copy(),
            "camera_pitch_deg": float(self.pitch[0].item()),
            "camera_distance": float(self.distances[0].item()),
            "applied_action": final_action[0].detach().cpu().numpy().copy(),  # (3,)
            "frame_bgr": self.last_rgb.copy(),               # HxWx3, uint8 (BGR)
            "amodal_mask": self.last_amodal_mask.copy(),     # HxW, uint8 (0/1)
            "visible_mask": self.last_visible_mask.copy(),   # HxW, uint8 (0/1)
            "occluded_mask": self.last_occluded_mask.copy(), # HxW, uint8 (0/1)
            "occ_ratio_val": float(self.curr_occ_ratio),
            "current_metric": current_metric_tensor.item()

        }

   
        return_values = (
            self.obs_buf.clone(),
            self.amodal_feat_buf.clone(),
            self.rew_buf.clone(),
            self.reset_buf.clone(),
            tsdf_vol_clone,  
            done_cr.clone(),
            current_metric_tensor_clone,
            reward_components,
            final_action,
            amodal_inference_time,
        )

      
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        return return_values


    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf.clone(), None

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        epi_per_scene_house = self.env_cfg.get("epi_per_scene_house", 50)
        if self.debug:
            print(f"[env:reset_idx] epi_count: {self._epi_count}")

        if self._epi_count % epi_per_scene_house == 0:
            if self.debug:
                print(f"[env:reset_idx] Scene Reset")
            global_index = self._epi_count // epi_per_scene_house
            scene_idx = global_index % len(self.scene_list)
            scene_num = self.scene_list[scene_idx]
            self.scene_init_azi = get_azimuth_for_scene(scene_num=scene_num)
            house_idx = (global_index // len(self.scene_list)) % len(self.house_list)
            house_num = self.house_list[house_idx]

            if self.inference:
                self.scene, self.camera, self.entities, self.cr_threshold = test_scene_build_house3k(
                    house_num=house_num, scene_num=scene_num, show_viewer=self.show_viewer, render=self.render
                )
                # self.scene, self.camera, self.entities, self.cr_threshold = test_scene_build_omniobject3d(
                #     house_num=house_num, scene_num=scene_num, show_viewer=self.show_viewer, render=self.render
                # )
            else:
                self.scene, self.camera, self.entities, self.cr_threshold = train_scene_build(
                    house_num=house_num, scene_num=scene_num, show_viewer=self.show_viewer, render=self.render
                )

            self.cr_threshold *= 0.95
            if self.debug:
                print(f"[env] CR_max95% : {self.cr_threshold}")

            self.obstacles_2d = load_obstacles_2d(
                filename=f'/scene_aabb/obb_obstacles_{scene_num}.json'
            )


            # amodal weights
            self.amodal_cfg['model_path'] ='/amodal_model/model_all_scene.pth'
            self.args = self.amodal_cfg
            self.predictor, self.fg_model, self.W, self.H, self.cfg = init_UOIAS(self.amodal_cfg)

            # GT mesh
            self.target_object = self.entities[0]
            self.mesh_gt = entity_to_trimesh(self.target_object)
            if self.mesh_gt is None:
                print("[!] house mesh is empty!")
                return

            # TSDF grid bounds
            min_xyz, max_xyz = self.mesh_gt.bounds
            size_x = max_xyz[0] - min_xyz[0]
            size_y = max_xyz[1] - min_xyz[1]
            size_z = max_xyz[2] - min_xyz[2]
            side = max(size_x, size_y, size_z)
            self.vol_bnds = np.array([
                [min_xyz[0], min_xyz[0] + side],
                [min_xyz[1], min_xyz[1] + side],
                [min_xyz[2], min_xyz[2] + side]
            ], dtype=float)

            Nx = self.Nx
            self.voxel_size = side / Nx
            self.margin = 0.5 * self.voxel_size
            self.tsdf_vol = TSDFVolume(self.vol_bnds, voxel_size=self.voxel_size, Nx=self.Nx)

        self._epi_count += 1

        # init state
        self.episode_max_steps_buf[env_ids] = 0
        el = 0.4
        init_dist = 25.0
        az0_deg   = self.scene_init_azi
        r0 = init_dist * math.cos(el)
        x0 = r0 * math.cos(math.radians(az0_deg))
        y0 = r0 * math.sin(math.radians(az0_deg))
        self.pos_xy[env_ids, 0] = x0
        self.pos_xy[env_ids, 1] = y0
     
        self.prev_pos_xy[env_ids, 0] = x0
        self.prev_pos_xy[env_ids, 1] = y0
        self.pitch[env_ids] = 0.0
        self.distances[env_ids] = init_dist 
        self._update_camera_pose(env_ids)

        # first frame
        _, rgb_img, depth_img, seg_img = self._get_frames()

        # amodal init
        rgb_for_amodal = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        with torch.cuda.amp.autocast():
            preds, pred_visibles, amodal_feature, amodal_feature_2, pred_classes, scores= infer_UOIAS_light(
                rgb_for_amodal, depth_img, self.predictor, self.fg_model, self.W, self.H, self.cfg, self.args
            )

        preds = preds.numpy()
        pred_visibles = pred_visibles.detach().cpu().numpy()
        amodal_feature = amodal_feature_2.detach().cpu().numpy()
        pred_classes = pred_classes.detach().cpu().numpy() if pred_classes is not None else None
        scores = scores.detach().cpu().numpy() if scores is not None else None
        
        
        target_indices = np.where(pred_classes == 0)[0]
        if len(target_indices) == 0:
            occ_ratio = 1.0
            target_amodal_mask = np.zeros_like(depth_img, dtype=np.uint8)
            target_visible_mask = np.zeros_like(depth_img, dtype=np.uint8)
            target_occluded_mask = np.zeros_like(depth_img, dtype=np.uint8)
        else:
            best_idx = target_indices[np.argmax(scores[target_indices])]
            target_amodal_mask = preds[best_idx]
            target_visible_mask = pred_visibles[best_idx]
            target_occluded_mask = target_amodal_mask & ~target_visible_mask
            kernel = np.ones((3, 3), np.uint8)
            target_occluded_mask = cv2.morphologyEx(
                target_occluded_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=3
            )
            occ_ratio = np.sum(target_occluded_mask) / (np.sum(target_amodal_mask) + 1e-10)
        self.curr_occ_ratio = occ_ratio
        self.last_occ_ratio = occ_ratio
        
        self.last_rgb = rgb_img.copy()
        self.last_amodal_mask = target_amodal_mask.astype(np.uint8)
        self.last_visible_mask = target_visible_mask.astype(np.uint8)
        self.last_occluded_mask = target_occluded_mask.astype(np.uint8)

        # buffers
        for i in env_ids:
            self.obs_buf[i].fill_(0.0)

        amodal_feature_tensor = torch.tensor(amodal_feature, dtype=torch.float32).to(self.device)
        amodal_feat_vec = self._pool_and_select_topk(amodal_feature_tensor, scores, K=self.amodal_max_instance)
        for i in env_ids:
            for st in range(self.k):
                self.amodal_feat_buf[i, st, :] = amodal_feat_vec

        # TSDF init + first integrate
        # depth noise 
        self.tsdf_vol.reset()
        depth_img_noised = depth_img
        mask = (seg_img == self.house_id)
        depth_img_noised[depth_img_noised >= 40] = 0
        depth_img_noised[~mask] = 0
        intr = self.camera.intrinsics
        extr = np.linalg.inv(self.camera.extrinsics)
        noise_cfg = None
        if self.depth_noise:
            noise_cfg = dict(
                seed=42,
                base_std=1.0/3.0,
                noise_start=6.0,
                skip_ge=40.0,
                debug_imshow=False,  
            )
        self.tsdf_vol.integrate(rgb_img, depth_img_noised, intr, extr, obs_weight=1.0, noise_cfg=noise_cfg)

        mesh_rec = mesh_from_tsdf_safe_colored(self.tsdf_vol, min_weight=0.5)
        if (mesh_rec is None) or (mesh_rec.faces.size == 0):
            init_cr_val = 0.0
        else:
            _, init_cr_val, _, _ = precision_recall_fscore_chamfer(
                self.mesh_gt, mesh_rec, tau=self.voxel_size, n_samples=None
            )
        self.init_cr = init_cr_val
        self.init_cr_tensor = torch.full((self.num_envs,), init_cr_val, device=self.device)
        self.prev_metric = self.init_cr_tensor.clone()

        # near_target init
        pos_now = np.array(self.camera.pos, dtype=np.float32)
        self.prev_range = float(np.linalg.norm(pos_now[:2]))

        ####
        self.max_cr_target_tensor = torch.full((self.num_envs,), float(self.cr_threshold), device=self.device)
        self.cr_headroom_tensor = (self.max_cr_target_tensor - self.init_cr_tensor).clamp_min(1e-4)

        self.prev_metric = self.init_cr_tensor.clone()


    def _update_camera_pose(self, env_ids=None):
        if env_ids is None:
            env_ids = range(self.num_envs)
        else:
    
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.tolist()

        el = 0.4
        for i in env_ids:
            x = self.pos_xy[i, 0].item()
            y = self.pos_xy[i, 1].item()
            r = math.sqrt(x * x + y * y)

            pos = np.array([x, y, 0.4], dtype=np.float32)
            look = self.compute_lookat(pos=pos, pitch_deg=self.pitch[i].item())
            self.camera.set_pose(pos=pos, lookat=look)

    def compute_lookat(self, pos: np.ndarray, pitch_deg: float):
        r = np.linalg.norm(pos[:2])
        pitch_rad = math.radians(pitch_deg)
        z = pos[2] + math.tan(pitch_rad) * r
        return np.array([0.0, 0.0, z], dtype=np.float32)

    def _get_frames(self):
        rgb_img, depth_img, seg_img, _ = self.camera.render(rgb=True, depth=True, segmentation=True)
        rgb_resized = cv2.resize(rgb_img, (84, 84))
        gray_array = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)
        gray_tensor = torch.from_numpy(gray_array).float().unsqueeze(0).to(self.device)
        return gray_tensor, rgb_img, depth_img, seg_img

    def _pool_and_select_topk(self, amodal_feature, scores, K=3):
        scores = torch.tensor(scores, device=self.device) if scores is not None else None
        if amodal_feature.shape[0] == 0:
            combined = torch.zeros((K, 256), device=self.device)
        else:
            pooled = amodal_feature.amax(dim=(2, 3))
            N = pooled.shape[0]
            if scores is None:
                selected = pooled[:min(N, K)]
                combined = torch.zeros((K, 256), device=self.device); combined[:selected.shape[0]] = selected
            elif N < K:
                _, idxs = torch.topk(scores, N); selected_feat = pooled[idxs]
                combined = torch.zeros((K, 256), device=self.device); combined[:N] = selected_feat
            else:
                _, idxs = torch.topk(scores, K); combined = pooled[idxs]
        return combined.view(-1)   

    @torch.no_grad()
    def sample_candidates_xy(
        self,
        K=10,
        P=5,  
        env_id=0,
        pitch_mode='fixed',  
        per_slot_trials=32,      
        uniform_anchor='random',  
        dedup_eps=1e-3,          
        allow_pitch_only_fallback=True   
    ):
   
        R = float(self.env_cfg.get(
            'circle_step_radius',
            max(abs(self.env_cfg['distance_scale'][0]),
                abs(self.env_cfg['distance_scale'][1]))
        ))
        el = 0.4
        xy0 = self.pos_xy[env_id].detach().cpu().numpy()
        r_min = self.distance_range[0] * math.cos(el)
        r_max = self.distance_range[1] * math.cos(el)

        def pitch_delta_set(P):
            curr_deg = float(self.pitch[env_id].item())
            if pitch_mode == 'fixed':
                abs_list = np.array([-30,-15, 0.0,15, 30.0][:P], dtype=np.float32)
            elif pitch_mode == 'random':
                lo_abs, hi_abs = 0.0, 50.0
                abs_list = np.random.uniform(lo_abs, hi_abs, size=P)
            elif pitch_mode == 'grid':
                lo_abs, hi_abs = 0.0, 50.0
                abs_list = np.linspace(lo_abs, hi_abs, P)
            else:  # 'centered'
                zmin, zmax = self.mesh_gt.bounds[0][2], self.mesh_gt.bounds[1][2]
                zc = 0.5 * (zmin + zmax)
                r = float(np.linalg.norm(xy0))
                center = math.degrees(math.atan2(zc - 0.4, r))
                center = float(np.clip(center, lo_abs, hi_abs))
                base = np.array([0, +7, -7, +14, -14], dtype=np.float32)
                abs_list = center + base[:P]
                abs_list = np.clip(abs_list, lo_abs, hi_abs)
            ddeg = np.clip(abs_list - curr_deg, self.pitch_delta_min, self.pitch_delta_max)
            return np.radians(ddeg).astype(np.float32)

        dp_set = pitch_delta_set(P)

        def find_valid_arc_segments():
            """Find collision-free arc segments on a circle of radius R."""
            n_samples = 10   # check every 36 degrees (sufficient in practice)
            angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
            valid_mask = np.zeros(n_samples, dtype=bool)
            prev_blocked_mask = np.zeros(n_samples, dtype=bool)  # candidates that are near the previous position
            
            # Previous position (used to avoid ping-ponging)
            prev_xy = self.prev_pos_xy[env_id].detach().cpu().numpy()
            min_distance = R * 0.7  # exclude moves closer than 70% of R to the previous position
            
            for i, theta in enumerate(angles):
                dx, dy = R * np.cos(theta), R * np.sin(theta)
                new_xy = xy0 + np.array([dx, dy], dtype=np.float32)
                
                # Check radial distance constraint
                r1 = float(np.linalg.norm(new_xy))
                if not (r_min <= r1 <= r_max):
                    continue
                
                # Collision check
                if self._would_collide_xy(xy0, new_xy):
                    continue
                
                # Check whether this is near the previous position (do not discard yet)
                is_near_prev = np.linalg.norm(new_xy - prev_xy) < min_distance
                if is_near_prev:
                    prev_blocked_mask[i] = True  # mark as near-previous candidate only
                else:
                    valid_mask[i] = True  # valid candidate that is not near the previous position
            
            # If there are candidates not near the previous position, use only those
            if np.any(valid_mask):
                return angles[valid_mask]
            # If everything is blocked, allow near-previous candidates as a fallback
            else:
                return angles[prev_blocked_mask]

        valid_angles = find_valid_arc_segments()

        # If we have valid angles, select up to K of them, roughly evenly spaced
        xy_list = []
        if len(valid_angles) > 0:
            if len(valid_angles) >= K:
                # If there are many angles, subsample them evenly
                indices = np.linspace(0, len(valid_angles) - 1, K, dtype=int)
                selected_angles = valid_angles[indices]
            else:
                # Otherwise, just use all valid angles
                selected_angles = valid_angles
            
            for theta in selected_angles:
                dx, dy = R * np.cos(theta), R * np.sin(theta)
                new_xy = xy0 + np.array([dx, dy], dtype=np.float32)
                xy_list.append(new_xy)

        # 3) Build final candidate set: for each XY, attach P pitch deltas → (K*P, 3)
        cand = []
        if len(xy_list) == 0 and allow_pitch_only_fallback:
            # If we truly cannot create any candidate, fall back to pitch-only moves
            for dp in dp_set:
                cand.append(np.array([0.0, 0.0, dp], dtype=np.float32))
            if len(cand) == 0:
                cand = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        else:
            for new_xy in xy_list:
                dxy = (new_xy - xy0).astype(np.float32)
                for dp in dp_set:
                    cand.append(np.array([dxy[0], dxy[1], dp], dtype=np.float32))

        # 4) Ensure we have K*P candidates: if not enough, pad with pitch-only moves (optional)
        need = K * P
        if len(cand) < need and allow_pitch_only_fallback:
            i = 0
            while len(cand) < need and i < 2 * need:
                cand.append(np.array([0.0, 0.0, dp_set[i % len(dp_set)]], dtype=np.float32))
                i += 1

        return np.stack(cand[:need], axis=0).astype(np.float32)


    def _would_collide_xy(self, old_xy, new_xy):
        collided, _ = line_segment_colliding_with_obstacles_2d_obb(old_xy, new_xy, self.obstacles_2d)
        return bool(collided)

    def _phi(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(y, 0.0, 1.0)
        if self.recon_shape == "power":
            return torch.pow(y, self.recon_gamma)  # φ(0)=0, φ(1)=1
        elif self.recon_shape == "exp":
            beta = self.recon_beta
            num = torch.exp(beta * y) - 1.0
            den = math.exp(beta) - 1.0 + 1e-8
            return num / den                        # φ(0)=0, φ(1)=1
        else:  # "linear"
            return y
        

    # ---------------- rewards ----------------
    def _reward_3d_recon(self):
        mesh_rec = mesh_from_tsdf_safe_colored(self.tsdf_vol, min_weight=0.5)
        if (mesh_rec is None) or (mesh_rec.faces.size == 0):
            current_metric = 0.0
        else:
            _, rec, _, _ = precision_recall_fscore_chamfer(
                self.mesh_gt, mesh_rec, tau=self.voxel_size, n_samples=None
            )
            current_metric = rec

        current_metric_tensor = torch.full((self.num_envs,), current_metric, device=self.device)

        denom = self.cr_headroom_tensor
        y_t   = torch.clamp((current_metric_tensor - self.init_cr_tensor) / denom, 0.0, 1.0)
        y_tm1 = torch.clamp((self.prev_metric      - self.init_cr_tensor) / denom, 0.0, 1.0)

    
        phi_t   = self._phi(y_t)
        phi_tm1 = self._phi(y_tm1)
        rew = (phi_t - phi_tm1) * self.recon_total_target

        self.prev_metric = current_metric_tensor
        return rew, current_metric_tensor

    def _reward_near_target(self):
        pos = np.array(self.camera.pos, dtype=np.float32)
        d = float(np.linalg.norm(pos[:2]))
        prev = d if (self.prev_range is None) else self.prev_range

        d_safe, band = 7.0, 0.5
        w_app, w_pen = 1.0, 0.5

       
        def U(x): return w_pen * max(0.0, d_safe - x)
        dU = U(d) - U(prev) 

        if d >= d_safe + band:
    
            approach_term = w_app * (prev - d)  
            shaped = approach_term
        elif d <= d_safe - band:
       
            shaped = - dU
        else:

            shaped = 0.0
				
        self.prev_range = d
        shaped = self.near_target_weight * shaped
        return torch.tensor(shaped, device=self.device)

    def _reward_step_cost(self):
        half = int(self.env_cfg['episode_max_steps'] * 0.5)
        mask = (self.episode_max_steps_buf >= half)
        step_cost = self.step_cost
        return torch.where(mask, torch.full((self.num_envs,), step_cost, device=self.device),
                                torch.zeros((self.num_envs,), device=self.device))

    def _reward_occ_ratio(self):
        delta = (self.curr_occ_ratio - self.last_occ_ratio) * self.occ_ratio_reward_weight
        self.last_occ_ratio = self.curr_occ_ratio
        target_visibility = self.target_visibility_weight * (1 - self.curr_occ_ratio)
        if self.debug:
            print(f"Occ ratio reward: {-delta + target_visibility:.4f}")
        return -delta + target_visibility
