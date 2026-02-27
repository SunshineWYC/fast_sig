from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render
import torch
import random

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss


def sampling_cameras(my_viewpoint_stack):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''

    num_cams = 10
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    
    return camlist


def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm
def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score_fastgs(camlist, gaussians, dataset, opt, pipe, bg, DENSIFY=False, pose_active=True):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: current Gaussian representation (model/state) used for rendering.
        pipe: rendering pipeline/context required by `render`.
        bg: background used for rendering.
        args: runtime config containing thresholds (e.g. `loss_thresh`).
        DENSIFY (bool): whether to compute and return the importance score
            used for densification. If False, only the pruning score is computed.

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        view_cam = camlist[view]
        gt_image = view_cam.original_image.cuda()
        render_image = render(
            view_cam,
            gaussians,
            pipe,
            bg,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            render_size=gt_image.shape[-2:],
            use_trained_exp=dataset.train_test_exp,
            retain_grad=False,
            cam_rot_delta=view_cam.cam_rot_delta if pose_active else None,
            cam_trans_delta=view_cam.cam_trans_delta if pose_active else None,
        )["render"]
        photometric_loss = compute_photometric_loss(view_cam, render_image)

        gt_image = view_cam.original_image.cuda()
        get_flag = True
        l1_loss_norm = get_loss(render_image, gt_image)
        
        metric_map = (l1_loss_norm > opt.loss_thresh).int().contiguous()

        render_pkg = render(
            view_cam,
            gaussians,
            pipe,
            bg,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            render_size=gt_image.shape[-2:],
            use_trained_exp=dataset.train_test_exp,
            cam_rot_delta=view_cam.cam_rot_delta if pose_active else None,
            cam_trans_delta=view_cam.cam_trans_delta if pose_active else None,
            retain_grad=False,
            get_flag=get_flag,
            metric_map=metric_map,
        )

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score) + 1e-6)
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score
