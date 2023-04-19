import tensorflow as tf
import time
from point_sample import point_sample, point_sample_slow, point_sample_2

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, ..., 1) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, ..., 1) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    # assert logits.shape[1] == 1
    # Tensorflow has Channels at last dim
    assert logits.shape[-1] == 1
    # gt_class_logits = logits.clone
    gt_class_logits = tf.identity(logits)
    return -(tf.abs(gt_class_logits))


def get_uncertain_point_coords_with_randomness(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, Hmask, Wmask, C) or (N, Hmask, Wmask, 1) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, P, C) or (N, P, 1) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, P, 1).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    # point_coords shape (N, P, 2)
    point_coords = tf.random.uniform(shape=[num_boxes, num_sampled, 2])

    # point_logits shape (N, P, C)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = tf.math.top_k(point_uncertainties[:, :, 0], k=num_uncertain_points)[1]
    # PyTorch function uses Long tensor (int64)
    idx = tf.cast(idx, dtype=tf.int64)
    shift = num_sampled * tf.range(num_boxes, dtype=tf.int64)
    idx += shift[:, None]

    idx = tf.reshape(idx, [-1])
    point_coords = tf.reshape(point_coords, [-1, 2])
    point_coords = tf.gather(point_coords, idx)
    point_coords = tf.reshape(point_coords, [num_boxes, num_uncertain_points, 2])

    if num_random_points > 0:
        point_coords = tf.concat(
            [
                point_coords,
                tf.random.uniform(shape=[num_boxes, num_random_points, 2]),
            ],
            axis=1,
        )
    return point_coords


if __name__ == "__main__":

    tf.random.set_seed(0)

    # src_mask (N, H, W, C) C = 1
    # src_mask = tf.random.uniform(shape=[3, 15, 20, 1])

    # Basic (non memory) performance test setup
    # test 100 masks of size 256x256
    # sampling_ops = 0
    # full_ops = 0

    src_mask = tf.random.uniform(shape=[3, 200, 200, 1])
    tgt_mask = tf.random.uniform(shape=[3, 200, 200, 1])

    # Setup to match Mask2Former sampling params
    OVERSAMPLE_RATIO = 3.0
    IMPORTANCE_SAMPLE_RATIO = 0.75
    NUM_POINTS = 100 * 100
    # NUM_POINTS = 112 * 112
    # NUM_POINTS = 15

    #print("uncertainty pts")
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
    #print("starting placeholder losses")
    lap2 = time.process_time()
    loss_placeholder_sampling = tf.nn.sigmoid_cross_entropy_with_logits(labels=point_labels, logits=point_logits)
    lap3 = time.process_time()
    loss_placeholder_full = tf.nn.sigmoid_cross_entropy_with_logits(labels=tgt_mask, logits=src_mask)
    lap4 = time.process_time()
    #print("done")

    get_coords_time = lap1 - start_time
    point_sample_time = lap2 - lap1
    sample_loss_time = lap3 - lap2
    full_time = lap4 - lap3

    print("Time to get point coords: {:.10f}".format(get_coords_time))
    print("Time to sample: {:.10f}".format(point_sample_time))
    print("Sampling loss time: {:.10f}".format(sample_loss_time))
    print("Full loss time: {:.10f}".format(full_time))

    # print(point_coords.shape)
    # print(point_coords)

    # assert point_coords.shape[0] == src_mask.shape[0]
    # assert point_coords.shape[1] == NUM_POINTS
    # assert point_coords.shape[2] == 2
