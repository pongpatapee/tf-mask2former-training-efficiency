import tensorflow as tf


def safe_get(input, n, x, y, c):
    """
    Provide zero padding for getting values from out of bounds indices from the 'input' matrix
    """
    # input shape = [N, H, W, C]
    H = input.shape[1]
    W = input.shape[2]

    return input[n, y, x, c] if (x >= 0 and x < W and y >= 0 and y < H) else 0

def safe_get_all(input, X, Y, H, W):
    """
    Provide zero padding for getting values from out of bounds indices from the 'input' matrix
    """
    # input shape = [N, H, W, C]

    # Make sure everything is in bounds
    mask = tf.math.logical_and(
    tf.math.logical_and(tf.math.less(X, W), tf.math.less(Y, H)),
    tf.math.logical_and(tf.math.greater_equal(X, 0), tf.math.greater_equal(Y, 0))
    )

    # indices = tf.stack([X, Y], axis=-1)
    indices = tf.stack([(X+1) * (Y+1)], axis=-1)
    print("")
    # print("type: ", indices.dtype)
    print("")

    # Use mask to conditionally gather values from input or return 0
    values = tf.where(mask, tf.gather_nd(input, tf.cast(indices, dtype=tf.int32)), 0)
    return values


def point_sample(input, point_coords, align_corners=False, **kwargs):
    """
    A TensorFlow implementation of point_sample from detectron. Sample points
    using bilinear interpolation from `point_coords` which is assumed to be a
    [0, 1] x [0, 1] square. Default mode for align_corners is False
    Args:
        input (Tensor): A tensor of shape (N, H, W, C) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1]
        normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, P, C) containing features for points
        in `point_coords`. The features are obtained via bilinear interpolation from
        `input`.
    """

    # assert correct dimensions
    assert len(input.shape) == 4
    assert len(point_coords.shape) == 3

    N = input.shape[0]
    H = input.shape[1]
    W = input.shape[2]
    C = input.shape[3]

    P = point_coords.shape[1]

    # changing x,y range from [0, 1] to [-1, 1]
    point_coords = 2 * point_coords - 1

    output = tf.zeros([N, P, C])
    
    # Flatten input
    input_flat = tf.reshape(input, [N * H * W * C])

    # Get X, a grid of values for (n, p, 0), and Y, a grid for (n, p, 1)
    point_coords_flat = tf.reshape(point_coords, [N * P, 2])
    x = point_coords_flat[:, 0]
    y = point_coords_flat[:, 1]

    X = tf.reshape(x, [N, P])
    Y = tf.reshape(y, [N, P])

    if align_corners:
        X = ((X + 1) / 2) * (W - 1)
        Y = ((Y + 1) / 2) * (H - 1)
    else:
        X = ((X + 1) * W - 1) / 2
        Y = ((Y + 1) * H - 1) / 2

    # for every (n, p, c) combination, you want to get NW, SW, NE, SE vals for interpolation
    # X has (n, p, 0), Y has (n, p, 1)

    # example: nw_val = safe_get(input, n, x1, y1, c)
    # x1, y1 = int(tf.floor(x)), int(tf.floor(y))
    # x2, y2 = x1 + 1, y1 + 1

    X_1, Y_1 = tf.floor(X), tf.floor(Y)
    X_2, Y_2 = X_1 + 1, Y_1 + 1

    Xn = X.numpy()
    Yn = Y.numpy()
    X_1n = X_1.numpy()
    X_2n = X_2.numpy()
    Y_1n = Y_1.numpy()
    Y_2n = Y_2.numpy()

    # Xn = tf.cast(X, dtype=tf.int32)
    # Yn = tf.cast(Y, dtype=tf.int32)
    # X_1n = tf.cast(X_1, dtype=tf.int32)
    # X_2n = tf.cast(X_2, dtype=tf.int32)
    # Y_1n = tf.cast(Y_1, dtype=tf.int32)
    # Y_2n = tf.cast(Y_2, dtype=tf.int32)

    # Xn = tf.cast(X, dtype=tf.int32).numpy()
    # Yn = tf.cast(Y, dtype=tf.int32).numpy()
    # X_1n = tf.cast(X_1, dtype=tf.int32).numpy()
    # X_2n = tf.cast(X_2, dtype=tf.int32).numpy()
    # Y_1n = tf.cast(Y_1, dtype=tf.int32).numpy()
    # Y_2n = tf.cast(Y_2, dtype=tf.int32).numpy()

    NW_VALS = safe_get_all(input_flat, X_1n, Y_1n, H, W)
    SW_VALS = safe_get_all(input_flat, X_1n, Y_2n, H, W)
    NE_VALS = safe_get_all(input_flat, X_2n, Y_1n, H, W)
    SE_VALS = safe_get_all(input_flat, X_2n, Y_2n, H, W)

    R1 = (((X_2n - Xn) / (X_2n - X_1n)) * NW_VALS +
          ((Xn - X_1n) / (X_2n - X_1n)) * NE_VALS)
    R2 = (((X_2n - Xn) / (X_2n - X_1n)) * SW_VALS +
          ((Xn - X_1n) / (X_2n - X_1n)) * SE_VALS)

    SAMPLED_PTS = (((Y_2n - Yn) / (Y_2n - Y_1n)) * R1 + 
                     ((Yn - Y_1n) / (Y_2n - Y_1n)) * R2)
    
    SAMPLED_PTS = tf.reshape(SAMPLED_PTS, output.shape)
    print("")
    print("type: ", SAMPLED_PTS.dtype)
    print("")
    

    output = tf.tensor_scatter_nd_add(output, indices=tf.cast(point_coords, dtype=tf.int32), updates=SAMPLED_PTS)
    return output

def point_sample_2(input, point_coords, align_corners=False, **kwargs):
    """
    A TensorFlow implementation of point_sample from detectron. Sample points
    using bilinear interpolation from `point_coords` which is assumed to be a
    [0, 1] x [0, 1] square. Default mode for align_corners is False
    Args:
        input (Tensor): A tensor of shape (N, H, W, C) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1]
        normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, P, C) containing features for points
        in `point_coords`. The features are obtained via bilinear interpolation from
        `input`.
    """

    # assert correct dimensions
    assert len(input.shape) == 4
    assert len(point_coords.shape) == 3

    N = input.shape[0]
    H = input.shape[1]
    W = input.shape[2]
    C = input.shape[3]

    P = point_coords.shape[1]

    # changing x,y range from [0, 1] to [-1, 1]
    point_coords = 2 * point_coords - 1

    output = tf.zeros([N, P, C])
    
    # Flatten input
    input_flat = tf.reshape(input, [N, H * W * C])

    # Get X, a grid of values for (n, p, 0), and Y, a grid for (n, p, 1)
    point_coords_flat = tf.reshape(point_coords, [N * P, 2])
    x = point_coords_flat[:, 0]
    y = point_coords_flat[:, 1]

    X = tf.reshape(x, [N, P])
    Y = tf.reshape(y, [N, P])

    if align_corners:
        X = ((X + 1) / 2) * (W - 1)
        Y = ((Y + 1) / 2) * (H - 1)
    else:
        X = ((X + 1) * W - 1) / 2
        Y = ((Y + 1) * H - 1) / 2

    # for every (n, p, c) combination, you want to get NW, SW, NE, SE vals for interpolation
    # X has (n, p, 0), Y has (n, p, 1)

    # example: nw_val = safe_get(input, n, x1, y1, c)
    # x1, y1 = int(tf.floor(x)), int(tf.floor(y))
    # x2, y2 = x1 + 1, y1 + 1

    X_1, Y_1 = tf.floor(X), tf.floor(Y)
    X_2, Y_2 = X_1 + 1, Y_1 + 1

    Xn = tf.cast(X, dtype=tf.float32).numpy()
    Yn = tf.cast(Y, dtype=tf.float32).numpy()
    X_1n = tf.cast(X_1, dtype=tf.float32).numpy()
    X_2n = tf.cast(X_2, dtype=tf.float32).numpy()
    Y_1n = tf.cast(Y_1, dtype=tf.float32).numpy()
    Y_2n = tf.cast(Y_2, dtype=tf.float32).numpy()

    print("X1 shape: ", X_1.shape)
    print("Y1 shape: ", Y_1.shape)
    print("X2 shape: ", X_2.shape)
    print("Y2 shape: ", Y_2.shape)

    # attempt 2 changes
    for n in range(N):
        for p in range(P):
            for c in range(C):
                nw_val = safe_get(input, n, X_1n[n, p], Y_1n[n, p], c)
                sw_val = safe_get(input, n, X_1n[n, p], Y_2n[n, p], c)
                ne_val = safe_get(input, n, X_2n[n, p], Y_1n[n, p], c)
                se_val = safe_get(input, n, X_2n[n, p], Y_2n[n, p], c)

                R1 = ((X_2n[n, p] - Xn[n, p]) / (X_2n[n, p] - X_1n[n, p])) * nw_val + ((Xn[n, p] - X_1n[n, p]) / (X_2n[n, p] - X_1n[n, p])) * ne_val
                R2 = ((X_2n[n, p] - Xn[n, p]) / (X_2n[n, p] - X_1n[n, p])) * sw_val + ((Xn[n, p] - X_1n[n, p]) / (X_2n[n, p] - X_1n[n, p])) * se_val
                sampled_point = ((Y_2n[n, p] - Yn[n, p]) / (Y_2n[n, p] - Y_1n[n, p])) * R1 + (
                    (Yn[n, p] - Y_1n[n, p]) / (Y_2n[n, p] - Y_1n[n, p])
                ) * R2

                output = tf.tensor_scatter_nd_add(output, indices=[[n, p, c]], updates=[sampled_point])
    return output


def point_sample_slow(input, point_coords, align_corners=False, **kwargs):
    """
    A TensorFlow implementation of point_sample from detectron. Sample points
    using bilinear interpolation from `point_coords` which is assumed to be a
    [0, 1] x [0, 1] square. Default mode for align_corners is False
    Args:
        input (Tensor): A tensor of shape (N, H, W, C) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1]
        normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, P, C) containing features for points
        in `point_coords`. The features are obtained via bilinear interpolation from
        `input`.
    """

    # assert correct dimensions
    assert len(input.shape) == 4
    assert len(point_coords.shape) == 3

    N = input.shape[0]
    H = input.shape[1]
    W = input.shape[2]
    C = input.shape[3]

    P = point_coords.shape[1]

    # changing x,y range from [0, 1] to [-1, 1]
    point_coords = 2 * point_coords - 1

    output = tf.zeros([N, P, C])
    for n in range(N):
        for p in range(P):
            x, y = point_coords[n, p, :]

            if align_corners:
                # Unnormalize coords from [-1, 1] to [0, H - 1] & [0, W - 1]
                x = ((x + 1) / 2) * (W - 1)
                y = ((y + 1) / 2) * (H - 1)
            else:
                # Unnormalize coords from [-1, 1] to [-0.5, H - 0.5] & [-0.5, W - 0.5]
                x = ((x + 1) * W - 1) / 2
                y = ((y + 1) * H - 1) / 2

            x1, y1 = int(tf.floor(x)), int(tf.floor(y))
            x2, y2 = x1 + 1, y1 + 1

            for c in range(C):
                nw_val = safe_get(input, n, x1, y1, c)
                sw_val = safe_get(input, n, x1, y2, c)
                ne_val = safe_get(input, n, x2, y1, c)
                se_val = safe_get(input, n, x2, y2, c)

                R1 = ((x2 - x) / (x2 - x1)) * nw_val + ((x - x1) / (x2 - x1)) * ne_val
                R2 = ((x2 - x) / (x2 - x1)) * sw_val + ((x - x1) / (x2 - x1)) * se_val
                sampled_point = ((y2 - y) / (y2 - y1)) * R1 + (
                    (y - y1) / (y2 - y1)
                ) * R2

                # might be inefficiently if tensor_scatter_nd_update is creating a new matrix for each index update
                output = tf.tensor_scatter_nd_update(
                    output, indices=[[n, p, c]], updates=[sampled_point]
                )

    return output
