# Detectron's point_sample
from torch.nn import functional as F
import torch
import time
from typing import List, Optional

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    
    # print("point coordinates being passed in from wrapper function")
    # print(2.0 * point_coords - 1.0)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def calculate_uncertainty(logits):
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def get_uncertain_point_coords_with_randomness(coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio):
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

if __name__ == "__main__":

    torch.manual_seed(1)

    # src_mask (N, H, W, C) C = 1
    # src_mask = tf.random.uniform(shape=[3, 15, 20, 1])

    # Basic (non memory) performance test setup
    # test 100 masks of size 256x256
    # sampling_ops = 0
    # full_ops = 0

    src_mask = torch.rand(size=[3, 1, 200, 200])
    tgt_mask = torch.rand(size=[3, 1, 200, 200])

    # Setup to match Mask2Former sampling params
    OVERSAMPLE_RATIO = 3.0
    IMPORTANCE_SAMPLE_RATIO = 0.75
    NUM_POINTS = 100 * 100
    # NUM_POINTS = 112 * 112
    # NUM_POINTS = 15

    print("uncertainty pts")
    start_time = time.process_time()
    point_coords = get_uncertain_point_coords_with_randomness(
        coarse_logits=src_mask,
        uncertainty_func=(lambda logits: calculate_uncertainty(logits)),
        num_points=NUM_POINTS,
        oversample_ratio=OVERSAMPLE_RATIO,
        importance_sample_ratio=IMPORTANCE_SAMPLE_RATIO,
    )

    # Get point labels, point logits
    lap1 = time.process_time()
    point_labels = point_sample(tgt_mask, point_coords, align_corners=False)
    point_logits = point_sample(src_mask, point_coords, align_corners=False)

    # Placeholder loss
    print("starting placeholder losses")
    lap2 = time.process_time()
    loss_placeholder_sampling = F.binary_cross_entropy_with_logits(point_logits, point_labels, reduction='none')
    lap3 = time.process_time()
    loss_placeholder_full = F.binary_cross_entropy_with_logits(src_mask, tgt_mask, reduction='none')
    lap4 = time.process_time()
    print("done")

    get_coords_time = lap1 - start_time
    point_sample_time = lap2 - lap1
    sample_loss_time = lap3 - lap2
    full_time = lap4 - lap3

    print("Time to get point coords: {:.10f}".format(get_coords_time))
    print("Time to sample: {:.10f}".format(point_sample_time))
    print("Sampling loss time: {:.10f}".format(sample_loss_time))
    print("Full loss time: {:.10f}".format(full_time))
