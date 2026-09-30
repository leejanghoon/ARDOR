# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import cv2
import imageio
import random
import numpy as np
import torch
# from thop import profile, count_parameters
# constants
from detectron2.engine import DefaultPredictor

from adet.config import get_cfg
from adet.utils.visualizer import visualize_pred_amoda_occ
from adet.utils.post_process import detector_postprocess, DefaultPredictor

from utils import *
from foreground_segmentation.model import Context_Guided_Network


def visualize_amodal_features(rgb_img, amodal_feature, pred_classes=None, scores=None, top_k=3):
    """
    Visualize amodal features in a CAM-like style as an overlay on the RGB image.

    Args:
        rgb_img: RGB image (H, W, 3).
        amodal_feature: Amodal feature tensor (N, 256, 14, 14) where N is the number of instances.
        pred_classes: Predicted classes (N,), optional.
        scores: Confidence scores (N,), optional.
        top_k: Number of top-scoring instances to visualize.

    Returns:
        overlay_img: Image with feature maps overlaid on top of the RGB image.
    """
    if amodal_feature.shape[0] == 0:
        print("No amodal features to visualize")
        return rgb_img
    
    # Original RGB image size
    h, w = rgb_img.shape[:2]
    
    # Select top-k instances
    num_instances = min(top_k, amodal_feature.shape[0])
    if scores is not None:
        # Top-k by score
        top_indices = np.argsort(scores)[-num_instances:][::-1]
    else:
        # Simply take the first k instances
        top_indices = list(range(num_instances))
    
    # Copy image for overlay
    overlay_img = rgb_img.copy()
    
    for i, idx in enumerate(top_indices):
        # Take the feature map for this instance: (256, 14, 14)
        feature_map = amodal_feature[idx].cpu().numpy()
        
        # Average over channels → single activation map (14, 14)
        activation_map = np.mean(feature_map, axis=0)
        
        # Check and handle NaN/Inf
        if np.isnan(activation_map).any() or np.isinf(activation_map).any():
            print(f"[WARNING] NaN/Inf detected in activation_map for instance {idx}, skipping...")
            continue
            
        # Safe normalization to [0, 1]
        map_min = activation_map.min()
        map_max = activation_map.max()
        if map_max - map_min < 1e-8:  # Nearly constant
            activation_map = np.ones_like(activation_map) * 0.5
        else:
            activation_map = (activation_map - map_min) / (map_max - map_min)
        
        activation_map = activation_map.astype(np.float32)
        
        # Resize to original image size
        activation_map_resized = cv2.resize(activation_map, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply color map using OpenCV (different colormap per instance)
        activation_map_uint8 = (activation_map_resized * 255).astype(np.uint8)
        
        colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO]
        colormap_idx = i % len(colormaps)
        colored_map = cv2.applyColorMap(activation_map_uint8, colormaps[colormap_idx])
        
        # Alpha mask (stronger activation → stronger overlay); mild gamma correction for contrast
        alpha_mask = activation_map_resized
        alpha_mask = np.power(alpha_mask, 0.7)
        
        # Alpha blending
        for c in range(3):
            overlay_img[:, :, c] = (
                overlay_img[:, :, c] * (1 - alpha_mask * 0.6) +
                colored_map[:, :, c] * (alpha_mask * 0.6)
            ).astype(np.uint8)
        
        # Optionally draw class/score text
        if pred_classes is not None and scores is not None:
            class_id = pred_classes[idx]
            if class_id == 0:
                class_id_str = "Target(house)"
            elif class_id == 1:
                class_id_str = "Occluders"
            else:
                class_id_str = "Unknown"
            score = scores[idx]
            text = f"Inst{idx}: {class_id_str} ({score:.2f})"
            cv2.putText(
                overlay_img, text, (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
    
    return overlay_img


def visualize_individual_feature_channels(amodal_feature, instance_idx=0, top_channels=6):
    """
    Visualize individual feature channels for a specific instance (e.g., top-k channels by activation).

    Args:
        amodal_feature: Amodal feature tensor (N, 256, 14, 14).
        instance_idx: Index of the instance to visualize.
        top_channels: Number of most active channels to visualize.

    Returns:
        final_img: Combined visualization image (or None if out of range).
    """
    if instance_idx >= amodal_feature.shape[0]:
        print(f"Instance index {instance_idx} out of range")
        return None
    
    feature_map = amodal_feature[instance_idx].cpu().numpy()  # (256, 14, 14)
    
    # Channel importance: mean absolute activation per channel
    channel_importance = np.mean(np.abs(feature_map), axis=(1, 2))
    
    # Select top-k channels
    top_channel_indices = np.argsort(channel_importance)[-top_channels:][::-1]
    
    combined_channels = []
    
    for i, ch_idx in enumerate(top_channel_indices):
        channel_map = feature_map[ch_idx]  # (14, 14)
        
        # Check NaN/Inf
        if np.isnan(channel_map).any() or np.isinf(channel_map).any():
            print(f"[WARNING] NaN/Inf detected in channel {ch_idx}, filling with zeros...")
            channel_map = np.zeros_like(channel_map)
        
        # Safe normalization to [0, 255]
        ch_min = channel_map.min()
        ch_max = channel_map.max()
        if ch_max - ch_min < 1e-8:
            channel_map = np.ones_like(channel_map) * 127
        else:
            channel_map = (channel_map - ch_min) / (ch_max - ch_min)
        
        channel_map = (channel_map * 255).astype(np.uint8)
        
        # Upscale 14x14 → 140x140
        channel_resized = cv2.resize(channel_map, (140, 140), interpolation=cv2.INTER_NEAREST)
        
        # Apply HOT colormap
        channel_colored = cv2.applyColorMap(channel_resized, cv2.COLORMAP_HOT)
        
        # Add text with channel index and importance
        text = f"Ch{ch_idx}: {channel_importance[ch_idx]:.3f}"
        cv2.putText(
            channel_colored, text, (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        combined_channels.append(channel_colored)
    
    # Arrange into a 2x3 grid
    top_row = np.hstack(combined_channels[:3])
    bottom_row = np.hstack(combined_channels[3:6])
    
    # If fewer than 6 channels, fill the remaining with black tiles
    if len(combined_channels) < 6:
        missing = 6 - len(combined_channels)
        for _ in range(missing):
            empty_channel = np.zeros((140, 140, 3), dtype=np.uint8)
            combined_channels.append(empty_channel)
        bottom_row = np.hstack(combined_channels[3:6])
    
    full_visualization = np.vstack([top_row, bottom_row])
    
    # Add a title strip
    title_img = np.zeros((30, full_visualization.shape[1], 3), dtype=np.uint8)
    title_text = f"Amodal Feature Channels - Instance {instance_idx}"
    cv2.putText(
        title_img, title_text, (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
    )
    
    # Stack title and visualization
    final_img = np.vstack([title_img, full_visualization])
    
    cv2.imshow('Amodal Feature Channels', final_img)
    
    return final_img


############################################################
def init_UOIAS(args):
    # UOAIS-Net
    device = args['device']
    print(f"[init_amodal] device:", device)
    cfg = get_cfg()
    cfg.merge_from_file(args['config_file'])
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args["model_path"]
    print(f"[init_amodal] model_path:", args["model_path"])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # CG-Net (foreground segmentation)
    if args["use_cgnet"]:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args["cgnet_weight_path"]))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.to(device)
        fg_model.eval()
    else:
        fg_model = None
    return predictor, fg_model, W, H, cfg


def infer_UOIAS(rgb_img, depth_img, predictor, fg_model, W, H, cfg, args):
    """
    Run full UOAIS-Net inference (optionally with CG-Net foreground filtering).

    Args:
        rgb_img (np.ndarray): RGB image array (H, W, 3).
        depth_img (np.ndarray): Depth image array (H, W).
        predictor (DefaultPredictor): Detectron2 predictor.
        fg_model (torch.nn.Module): Pre-trained CG-Net model.
        W (int): Target width.
        H (int): Target height.
        cfg: Configuration object.
    """
    device = args["device"]
    rgb_img = cv2.resize(rgb_img, (W, H))

    min_val, max_val = cfg.INPUT.DEPTH_RANGE[0], cfg.INPUT.DEPTH_RANGE[1]
    depth_img = normalize_depth(depth_img, min_val=min_val, max_val=max_val)
    depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_img = inpaint_depth(depth_img)

    # UOAIS-Net input composition
    if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
        uoais_input = depth_img
    elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY:
        uoais_input = np.concatenate([rgb_img, depth_img], -1)

    outputs = predictor(uoais_input)
    instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

    # CG-Net inference for foreground filtering
    if args["use_cgnet"]:
        fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
        fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
        fg_depth_input = cv2.resize(depth_img, (320, 240))
        fg_depth_input = array_to_tensor(fg_depth_input[:, :, 0:1]).unsqueeze(0) / 255
        fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
        fg_output = fg_model(fg_input.to(device))
        fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
        fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
        fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Extract predictions
    preds = instances.pred_masks.detach().cpu().numpy()
    pred_visibles = instances.pred_visible_masks.detach().cpu().numpy()
    bboxes = instances.pred_boxes.tensor.detach().cpu().numpy()
    pred_occs = instances.pred_occlusions.detach().cpu().numpy()
    pred_classes = instances.pred_classes.detach().cpu().numpy()
    scores = instances.scores.detach().cpu().numpy()
    
    # Amodal features
    amodal_feature = predictor._amodal_feature.detach().cpu().numpy()

    # Foreground filtering using CG-Net result
    if args["use_cgnet"]:
        remove_idxs = []
        for i, pred_visible in enumerate(pred_visibles):
            iou = np.sum(np.bitwise_and(pred_visible, fg_output)) / np.sum(pred_visible)
            if iou < 0.5:
                remove_idxs.append(i)
        print(f"remove_idxs : {remove_idxs}")
        preds = np.delete(preds, remove_idxs, 0)
        pred_visibles = np.delete(pred_visibles, remove_idxs, 0)
        bboxes = np.delete(bboxes, remove_idxs, 0)
        pred_occs = np.delete(pred_occs, remove_idxs, 0)
        pred_classes = np.delete(pred_classes, remove_idxs, 0)
        scores = np.delete(scores, remove_idxs, 0)

    # Reorder predictions: occluded instances first
    idx_shuf = np.concatenate((np.where(pred_occs == 1)[0], np.where(pred_occs == 0)[0]))
    preds, pred_occs, bboxes, pred_visibles, scores, pred_classes = \
        preds[idx_shuf], pred_occs[idx_shuf], bboxes[idx_shuf], pred_visibles[idx_shuf], scores[idx_shuf], pred_classes[idx_shuf]
    amodal_feature = amodal_feature[idx_shuf]

    # Optional built-in visualization
    visualize = False
    if visualize:
        vis_img = visualize_pred_amoda_occ(rgb_img, preds, bboxes, pred_occs)
        vis_all_img = np.hstack([rgb_img, depth_img, vis_img])
    else:
        vis_img = None
        vis_all_img = None

    return preds, pred_visibles, bboxes, pred_occs, scores, pred_classes, vis_img, vis_all_img, amodal_feature


def infer_UOIAS_light(rgb_img, depth_img, predictor, fg_model, W, H, cfg, args):
    """
    Lightweight inference wrapper:
      - Minimal post-processing
      - Returns tensors instead of numpy when possible
      - Optional amodal feature visualization
    """
    # Preprocess inputs
    rgb_img = cv2.resize(rgb_img, (W, H))
    min_val, max_val = cfg.INPUT.DEPTH_RANGE[0], cfg.INPUT.DEPTH_RANGE[1]
    depth_img = normalize_depth(depth_img, min_val=min_val, max_val=max_val)
    depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_img = inpaint_depth(depth_img)

    # UOAIS-Net input
    if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
        uoais_input = depth_img
    elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY:
        uoais_input = np.concatenate([rgb_img, depth_img], -1)

    outputs = predictor(uoais_input)
    instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

    # CG-Net inference (skipped entirely if not needed)
    if args.get("use_cgnet", False):
        # CG-Net related code could be added here if needed
        pass

    # Extract only necessary values; keep as tensors when possible
    preds = instances.pred_masks                     # (N, H, W), torch.Tensor
    pred_visibles = instances.pred_visible_masks     # (N, H, W), torch.Tensor
    amodal_feature = predictor._amodal_feature       # feature from occlusion guidance layer
    amodal_feature_2 = predictor._amodal_feature_2   # feature from amodal mask head
    pred_classes = instances.pred_classes if hasattr(instances, 'pred_classes') else None
    scores = instances.scores if hasattr(instances, 'scores') else None

    # Optional amodal feature visualization
    enable_amodal_vis = args.get("enable_amodal_vis", True)
    
    if enable_amodal_vis and amodal_feature.shape[0] > 0:
        try:
            pred_classes_np = pred_classes.detach().cpu().numpy() if pred_classes is not None else None
            scores_np = scores.detach().cpu().numpy() if scores is not None else None

            # (All overlay displays are currently commented out to avoid side effects)
            # You can re-enable visualize_amodal_features / visualize_individual_feature_channels here.

            cv2.waitKey(1)
            cv2.waitKey(1)
        except Exception as e:
            print("")
            # print(f"[AMODAL VIS ERROR] {str(e)}")
            # print("[AMODAL VIS ERROR] Skipping visualization...")
    

    # Built-in visualization from original code is disabled; we use our custom visualization instead.
    visualize = False
    if visualize:
        preds_np = preds.detach().cpu().numpy()
        bboxes = instances.pred_boxes.tensor.detach().cpu().numpy()
        pred_occs = instances.pred_occlusions.detach().cpu().numpy()
        vis_img = visualize_pred_amoda_occ(rgb_img, preds_np, bboxes, pred_occs)
        cv2.imshow('amodal-occluder predictions', vis_img)
        
        # Example: overlay only the top-1 scored instance mask
        if len(preds_np) > 0:
            scores_np = scores.detach().cpu().numpy() if scores is not None else None
            if scores_np is not None:
                top1_idx = np.argmax(scores_np)
                top1_mask = preds_np[top1_idx]
                top1_score = scores_np[top1_idx]
                
                overlay_img = rgb_img.copy()
                mask_colored = np.zeros_like(rgb_img)
                mask_colored[top1_mask > 0] = [0, 0, 255]
                
                alpha = 0.4
                overlay_img = cv2.addWeighted(overlay_img, 1 - alpha, mask_colored, alpha, 0)
                
                text = f"Top1 Instance (Score: {top1_score:.2f})"
                cv2.putText(
                    overlay_img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                cv2.imshow('Top1 Instance Mask', overlay_img)
        
        cv2.waitKey(1)
    else:
        vis_img = None
        vis_all_img = None

    return preds, pred_visibles, amodal_feature, amodal_feature_2, pred_classes, scores
