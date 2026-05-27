#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import random
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from torch import nn
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except ImportError:
    SparseGaussianAdam = None

logger = logging.getLogger(__name__)

class GaussianModel:
    """Gaussian Splatting model for 3D scene representation and rendering."""
    
    # Configuration constants for training
    DEFAULT_INIT_OPACITY = 0.1
    DEFAULT_SCALE_REDUCTION = 0.8
    DEFAULT_WINDOW_SIZE = 100
    WARMUP_STEPS = 100
    EMA_ALPHA = 0.1
    ALPHA_RANGE = (0.1, 2.0)
    COLOR_GRADIENT_WEIGHT = 20.0
    CLONE_COLOR_WEIGHT = 100.0
    MAX_OPACITY = 0.01
    PRUNING_EXTENT_FACTOR = 0.1
    HISTORY_WINDOW = 50
    LOSS_DECLINE_THRESHOLD = 0.05
    COOLDOWN_PERIOD = 10
    ADAPTIVE_FACTOR_INCREASE = 1.1
    ADAPTIVE_FACTOR_DECREASE = 0.9

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.color_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.window_size = self.DEFAULT_WINDOW_SIZE
        self.loss_history = []
        self.base_alpha_history = []
        self.adaptive_final = None
        self.final_adjust_counter = 0
        self.verbose = False

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.color_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        color_gradient_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.color_gradient_accum = color_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: List, spatial_lr_scale: float) -> None:
        """Initialize Gaussian model from point cloud.
        
        Args:
            pcd: Point cloud with xyz coordinates and rgb colors
            cam_infos: List of camera information objects
            spatial_lr_scale: Spatial learning rate scale factor
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        logger.info(f"Initialized with {fused_point_cloud.shape[0]} points from point cloud")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            self.DEFAULT_INIT_OPACITY * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            if SparseGaussianAdam is None:
                logger.warning("SparseGaussianAdam not available, falling back to standard Adam optimizer")
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            else:
                try:
                    self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
                except RuntimeError as e:
                    logger.warning(f"Failed to initialize SparseGaussianAdam: {e}. Using standard Adam.")
                    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration: int) -> Optional[float]:
        """Update learning rate for all optimizer groups based on scheduler.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Updated learning rate for xyz parameters, or None
        """
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        Prunes Gaussian points using a boolean mask.
        Keeps only points where mask == False.
        Handles missing tmp_radii and size mismatches safely.
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # Update model tensors
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Gradient accumulators
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.color_gradient_accum = self.color_gradient_accum[valid_points_mask]

        # --- Safe denom handling ---
        if self.denom.shape[0] != valid_points_mask.shape[0]:
            print(f"[Warning] denom size mismatch: denom={self.denom.shape[0]}, mask={valid_points_mask.shape[0]}")
            new_denom = torch.zeros((self._xyz.shape[0], 1), device=self.denom.device)
            copy_len = min(self.denom.shape[0], valid_points_mask.shape[0])
            new_denom[:copy_len] = self.denom[:copy_len]
            self.denom = new_denom

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # --- Safe tmp_radii handling ---
        # tmp_radii only exists during densification; skip if absent or None
        if hasattr(self, "tmp_radii") and self.tmp_radii is not None:
            if self.tmp_radii.shape[0] == valid_points_mask.shape[0]:
                self.tmp_radii = self.tmp_radii[valid_points_mask]
            else:
                print(f"[Warning] tmp_radii size mismatch: tmp_radii={self.tmp_radii.shape[0]}, mask={valid_points_mask.shape[0]}")
                self.tmp_radii = None

        # Ensure optimizer and GPU state are synchronized
        torch.cuda.synchronize()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        torch.cuda.synchronize()
        
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grads_color, grad_threshold, scene_extent, budget, color_ratio, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        import random
        if random.random() > color_ratio:
            padded_grad_color = torch.zeros((n_init_points), device="cuda")
            padded_grad_color[:grads_color.shape[0]] = grads_color.squeeze()
            padded_grad = padded_grad + padded_grad_color*20

        # breakpoint()
        

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        current_num = max(budget - self.get_xyz.shape[0], 0)

        if current_num <= 0:
            # Budget exhausted: keep the function flow unchanged, but block new selections.
            selected_pts_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        else:
            # Randomly keep up to `current_num` active candidates.
            selected_pts_mask = selected_pts_mask.clone().masked_scatter_(selected_pts_mask, torch.tensor([i < current_num for i in range(selected_pts_mask.sum().item())], dtype=torch.bool, device=selected_pts_mask.device)[torch.randperm(selected_pts_mask.sum().item(), device=selected_pts_mask.device)])

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)


        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grads_color, grad_threshold, scene_extent, budget, color_ratio):
        # Extract points that satisfy the gradient condition
        import random
        if random.random() > color_ratio:
            grads += grads_color*100

        

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        current_num = max((budget - self.get_xyz.shape[0]), 0)

        if current_num <= 0:
            # Budget exhausted: keep the function flow unchanged, but block new selections.
            selected_pts_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        else:
            # Randomly keep up to `current_num` active candidates.
            selected_pts_mask = selected_pts_mask.clone().masked_scatter_(selected_pts_mask, torch.tensor([i < current_num for i in range(selected_pts_mask.sum().item())], dtype=torch.bool, device=selected_pts_mask.device)[torch.randperm(selected_pts_mask.sum().item(), device=selected_pts_mask.device)])

        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, budget, color_ratio):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_color = self.color_gradient_accum / self.denom
        grads_color[grads_color.isnan()] = 0.0

        # print(f'grads: {grads.mean()}, {grads.max()}, {grads.min()}, grads_color: {grads_color.mean()}, {grads_color.max()}, {grads_color.min()}')

        self.tmp_radii = radii
        self.densify_and_clone(grads, grads_color, max_grad, extent, budget, color_ratio)
        self.densify_and_split(grads, grads_color, max_grad, extent, budget, color_ratio)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def compute_position_gradients(loss, gaussians):
        """
        Computes ∂loss/∂xyz for visualization without altering optimizer states.
        Returns grad magnitudes as a [N] tensor.
        """
        if not hasattr(gaussians, "_xyz"):
            raise ValueError("Gaussian model missing _xyz attribute.")

        xyz = gaussians._xyz
        if not xyz.requires_grad:
            xyz = xyz.clone().detach().requires_grad_(True)

        # Re-run the forward pass for grad computation
        loss.backward(retain_graph=True)

        if xyz.grad is None:
            # fallback: manually compute using autograd.grad
            grads = torch.autograd.grad(loss, xyz, retain_graph=True, allow_unused=True)[0]
        else:
            grads = xyz.grad

        if grads is None:
            raise RuntimeError("No position gradients available — maybe xyz not used in loss?")

        grad_mag = torch.norm(grads, dim=1)
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
        return grad_mag

    
    def prune_gaussians(self, percent, import_score: list):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, gaussian_color, update_filter, iteration):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # breakpoint()
        if len(gaussian_color.shape) == 3:
            self.color_gradient_accum[update_filter] += torch.norm(gaussian_color.grad[update_filter,0, :2], dim=-1, keepdim=True)
        else:
            self.color_gradient_accum[update_filter] += torch.norm(gaussian_color.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
            
    def calculate_budget(self, initial, final, steps, cur_step, mode='exp'):
        if mode == 'cosine':
            x = np.linspace(0, np.pi, steps)
            factor = (1-np.cos(x)) / 2
            values = initial + factor*(final-initial)
        
        elif mode == 'linear':
            if steps <= 1:
                return final
            t = np.linspace(0, 1, steps)
            values = initial + t * (final - initial)
        
        elif mode == 'exp':
            t = np.linspace(0, 1, steps)
            k = np.log(final / initial)
            values = initial * np.exp(k * t)
        
        elif mode == 'log':
            t = np.linspace(1, 100, steps)
            values = initial + (np.log(t) / np.log(100)) * (final - initial)
        elif mode == 'adaptive_power':
            t = np.linspace(1, 100, steps)
            window_size = self.window_size
            warmup_steps = 100
            ema_alpha = 0.1
            
            if len(self.loss_history) <= warmup_steps:
                alpha = 1.0
            else:
                valid_losses = self.loss_history[warmup_steps:]
                
                # EMA平滑
                ema_losses = []
                ema_value = valid_losses[0]
                for loss in valid_losses:
                    ema_value = ema_alpha * loss + (1 - ema_alpha) * ema_value
                    ema_losses.append(ema_value)
                ema_losses = np.array(ema_losses)
                
                # 计算累积base_alpha
                # 使用从训练开始到现在的所有数据来计算基础alpha
                all_steps = np.arange(1, len(ema_losses) + 1)
                log_steps = np.log(all_steps)
                log_losses = np.log(ema_losses)
                base_alpha, _ = np.polyfit(log_steps, log_losses, 1)
                base_alpha = abs(base_alpha)  # 取绝对值确保正值
                
                # 如果需要更平滑的base_alpha变化，可以使用历史平均
                if not hasattr(self, 'base_alpha_history'):
                    self.base_alpha_history = []
                self.base_alpha_history.append(base_alpha)
                history_window = min(50, len(self.base_alpha_history))  # 使用最近50个base_alpha
                base_alpha = np.mean(self.base_alpha_history[-history_window:])
                
                # 计算近期趋势偏差
                recent_window = min(window_size, len(ema_losses))
                recent_steps = np.arange(len(ema_losses) - recent_window + 1, len(ema_losses) + 1)
                log_recent_steps = np.log(recent_steps)
                log_recent_losses = np.log(ema_losses[-recent_window:])
                recent_slope, _ = np.polyfit(log_recent_steps, log_recent_losses, 1)
                
                # 偏差现在是相对于累积base_alpha的偏差
                deviation = recent_slope - base_alpha
                
                # 调整系数可以控制偏差的影响程度
                deviation_weight = 0.5  # 可以调整这个权重来控制适应性
                alpha = base_alpha + deviation_weight * np.tanh(deviation)
                
                # 限制范围
                alpha = np.clip(alpha, 0.1, 2.0)
                
                # 调试信息
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"Step {len(self.loss_history)}:")
                    print(f"Cumulative Base Alpha: {base_alpha:.4f}")
                    print(f"Recent Slope: {recent_slope:.4f}")
                    print(f"Deviation: {deviation:.4f}")
                    print(f"Final Alpha: {alpha:.4f}")
            
            values = initial + ((t**alpha - 1)/(100**alpha - 1)) * (final - initial)

        elif mode == 'adaptive_power_th':
            t = np.linspace(1, 100, steps)
            window_size = self.window_size
            warmup_steps = 100
            ema_alpha = 0.1
            
            if len(self.loss_history) <= warmup_steps:
                alpha = 1.0
                adaptive_final = final
            else:
                valid_losses = self.loss_history[warmup_steps:]
                
                # EMA平滑 loss
                ema_losses = []
                ema_value = valid_losses[0]
                for loss in valid_losses:
                    ema_value = ema_alpha * loss + (1 - ema_alpha) * ema_value
                    ema_losses.append(ema_value)
                ema_losses = np.array(ema_losses)
                
                # 计算累积base_alpha
                all_steps = np.arange(1, len(ema_losses) + 1)
                log_steps = np.log(all_steps)
                log_losses = np.log(ema_losses)
                base_alpha, _ = np.polyfit(log_steps, log_losses, 1)
                base_alpha = abs(base_alpha)
                
                # 平滑base_alpha
                if not hasattr(self, 'base_alpha_history'):
                    self.base_alpha_history = []
                self.base_alpha_history.append(base_alpha)
                history_window = min(50, len(self.base_alpha_history))
                base_alpha = np.mean(self.base_alpha_history[-history_window:])
                
                # 计算近期趋势
                recent_window = min(window_size, len(ema_losses))
                recent_steps = np.arange(len(ema_losses) - recent_window + 1, len(ema_losses) + 1)
                log_recent_steps = np.log(recent_steps)
                log_recent_losses = np.log(ema_losses[-recent_window:])
                recent_slope, _ = np.polyfit(log_recent_steps, log_recent_losses, 1)
                
                # 计算loss下降速率
                loss_decline_rate = -recent_slope  # 负斜率表示loss在下降
                
                # 动态调整final值
                if not hasattr(self, 'adaptive_final'):
                    self.adaptive_final = final
                    self.final_adjust_counter = 0
                
                # 根据loss下降速率调整final
                adjust_threshold = 0.05  # 触发调整的阈值
                cooldown_period = 10    # 调整之间的冷却期
                
                self.final_adjust_counter += 1
                
                if self.final_adjust_counter >= cooldown_period:
                    if loss_decline_rate > adjust_threshold:
                        # loss下降快，可以适当增加final
                        increase_factor = 1.1  # 增加10%
                        self.adaptive_final = min(self.adaptive_final * increase_factor, final * 1.5)  # 设置上限
                        self.final_adjust_counter = 0
                    elif loss_decline_rate < -adjust_threshold:
                        # loss下降变慢或上升，降低final
                        decrease_factor = 0.9  # 减少10%
                        self.adaptive_final = max(self.adaptive_final * decrease_factor, final * 0.5)  # 设置下限
                        self.final_adjust_counter = 0
                
                adaptive_final = self.adaptive_final
                
                # 计算alpha
                deviation = recent_slope - base_alpha
                deviation_weight = 0.5
                alpha = base_alpha + deviation_weight * np.tanh(deviation)
                alpha = np.clip(alpha, 0.1, 2.0)
                
                # 调试信息
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"Step {len(self.loss_history)}:")
                    print(f"Cumulative Base Alpha: {base_alpha:.4f}")
                    print(f"Recent Slope: {recent_slope:.4f}")
                    print(f"Loss Decline Rate: {loss_decline_rate:.4f}")
                    print(f"Current Adaptive Final: {adaptive_final}")
                    print(f"Final Alpha: {alpha:.4f}")
            
            values = initial + ((t**alpha - 1)/(100**alpha - 1)) * (adaptive_final - initial)
            
        elif mode == 'power':
            alpha = 2.0
            t = np.linspace(1, 100, steps)
            values = initial + ((t**alpha - 1**alpha)/(100**alpha - 1**alpha)) * (final - initial)
        
        elif mode == 'para':
            t = np.linspace(0, 1, steps)
            values = initial + (final - initial) * t**2
        
        elif mode == 'beizer':
            control_points=[0.2, 0.8]
            t_values = np.linspace(0, 1, steps)

            P0, P3 = initial, final
            P1, P2 = initial + (final - initial) * control_points[0], initial + (final - initial) * control_points[1]
            
            values = (1 - t_values)**3 * P0 + \
                    3 * (1 - t_values)**2 * t_values * P1 + \
                    3 * (1 - t_values) * t_values**2 * P2 + \
                    t_values**3 * P3

        return int(values[cur_step])