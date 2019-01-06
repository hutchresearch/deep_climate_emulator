import sys
import tensorflow as tf


def swish_activation(x):
    """
    Applies 'swish' activation. See: https://arxiv.org/abs/1710.05941

    Args:
        x (ndarray): Pre-activation input tensor

    Returns:
        (ndarray): Post-activation input tensor

    """
    return x * tf.sigmoid(x)


def gaussian_weight_init(shape, tn_stddev):
    """
    Initializes weight vector by drawing from a truncated normal distribution.

    Args:
        shape (array-like): Shape of the weight tensor
        tn_stddev (float): Standard deviation of the normal distribution
            before truncation.

    Returns:
        (tf.Variable): TensorFlow variable containing the initialized
            weight tensor.

    """
    return tf.Variable(tf.truncated_normal(shape, stddev=tn_stddev), name="W")


def residual_unit(prev_a, prev_lprime, layers, tn_stddev, projection):
    """
    Construct a "residual unit", which is a set of layers encompassed by a
    shortcut connection.

    Args:
        prev_a (ndarray): Output of the previous layer (mb x w x w x l)
        prev_lprime (int): Number of channels in the output of the previous
            layer
        layers (list): List of layer definitions
        tn_stddev (float): Standard deviation used for truncated normal
            weight initialization
        projection (bool): If true, shortcut with match dimensions with a
           linear projection. Else, will zero-pad to match dimensions.

    Returns:
        (ndarray): Output of residual unit (mb x w_prime x w_prime x l_prime).
        (int): Number of output channels.
    """
    # Save input and initial dimension to be used for the shortcut connection
    resinput = prev_a
    initial_dim = prev_lprime

    # Build CNN, putting each layer in its own scope
    layer_id = 1
    for layer in layers:
        with tf.variable_scope("layer" + str(layer_id) + "_" + layer["type"]):
            if layer["type"] == "conv":
                prev_a = tf.layers.batch_normalization(prev_a)
                if layer["activation"] == "swish":
                    prev_a = swish_activation(prev_a)
                else:
                    prev_a = tf.nn.relu(prev_a)
                prev_a = res2d(x=prev_a, k=layer["kernel_width"], l=prev_lprime,
                               l_prime=layer["filters"], s=layer["stride"],
                               p=layer["padding"], act=layer["activation"],
                               tn_stddev=tn_stddev)
                prev_lprime = layer["filters"]
            else:
                prev_a = max_pool(x=prev_a, k=layer["kernel_width"], s=layer["stride"])
        layer_id += 1

    final_dim = prev_lprime

    # Match dimensions and combine the output of the shortcut and residual unit
    add = match_dims(resinput, initial_dim, final_dim, projection)
    a = add + prev_a

    return a, final_dim


def match_dims(x, l, l_prime, projection):
    """
    Modifies the input tensor to match dimensions with the output of the
    residual unit.

    Args:
        x (ndarray): Input tensor of the residual unit
        l (int): Number of channels in the output of the previous layer
        l_prime (int): Number of channels in the input of the next layer
        projection (bool): If true, shortcut with match dimensions with a
           linear projection. Else, will zero-pad to match dimensions

    Returns:
        (ndarray): Output of the shortcut connection. This will be added to
        the output of the residual unit.
    """
    if projection:
        # Match dimension using a linear projection
        add = conv2d(x=x, k=3, l=l, l_prime=l_prime, s=1, p="SAME", act="identity")
    else:
        num_channels = x.get_shape().as_list()[3]
        if num_channels < l_prime:
            # Zero-pad to match dimensions
            paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, l_prime - num_channels]])
            add = tf.pad(x, paddings, "CONSTANT")
        elif num_channels > l_prime:
            # Slice to match dimensions
            add = tf.reshape(x, [-1, num_channels, 64, 128])
            add = tf.slice(add, [0, num_channels - l_prime, 0, 0], [-1, l_prime, 64, 128])
            add = tf.reshape(add, [-1, 64, 128, l_prime])
        else:
            add = x

    return add


def conv2d(x, k, l, l_prime, s, p, act):
    """
    Applies the convolution operation using tf.nn.conv2d (see:
    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)

    Args:
        x (ndarray): Input tensor with shape (mb x w x w x l)
        k (int): Kernel width (will be k x k, spatially)
        l (int): Number of input channels
        l_prime (int): Number of kernels to apply (i.e. number of output
            channels)
        s (int): Stride
        p (str): Padding (either "SAME" or "VALID")
        act (str): Hidden activation function. Must be in the set {"relu",
            "tanh", "identity", "swish"}

    Returns:
        (ndarray): Output of the convolutional layer with shape
            (mb x w_prime x w_prime x l_prime)
    """
    # Convolution weights
    W = tf.get_variable(name="W",
                        shape=(k, k, l, l_prime),
                        dtype=tf.float32,
                        initializer=tf.glorot_uniform_initializer())

    # Pick activation and modify bias constant, if needed
    b_const = 0.0
    if act == "relu":
        b_const = 0.1
        act = tf.nn.relu
    elif act == "tanh":
        act = tf.nn.tanh
    elif act == "identity":
        act = tf.identity
    elif act == "swish":
        act = swish_activation
    else:
        sys.exit("Error: Invalid activation function: %s" % act)

    # Bias weights
    b = tf.get_variable(name="b",
                        shape=l_prime,
                        initializer=tf.constant_initializer(b_const))
    
    # Apply convolution
    z = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=p)
    
    # Add bias
    z = z + b
    
    # Apply activation function
    z = act(z)
    
    return z


def res2d(x, k, l, l_prime, s, p, act, tn_stddev):
    """
    Adds a convolutional layer to the graph without applying the activation 
    function. The activation function will be applied after combining the
    hidden layer output with the shortcut connection output.
    
    Args:
        x (ndarray): Input tensor with shape (mb x w x w x l)
        k (int): Kernel width (will be k x k, spatially)
        l (int): Number of input channels
        l_prime (int): Number of kernels to apply (i.e. number of output
            channels)
        s (int): Stride
        p (str): Padding (either "SAME" or "VALID")
        act (str): Hidden activation function. Must be in the set {"relu",
            "tanh", "identity", "swish"}
        tn_stddev (float): Standard deviation used in truncated normal 
            initializer

    Returns:
        (ndarray): Output of convolutional layer, a tensor with shape 
            (mb x w_prime x w_prime x l_prime)

    """
    # Initialize convolution weights
    W = gaussian_weight_init([k, k, l, l_prime], tn_stddev)

    # Modify bias constant for ReLU
    b_const = 0.0
    if act == "relu":
        b_const = 0.1
    b = tf.Variable(tf.constant(b_const, shape=[l_prime]), name="b")

    # Apply convolution operation using tf.nn.conv2d
    z = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=p)

    return z + b


def max_pool(x, k, s):
    """
    Adds a max pooling layer. See:
    https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    
    Args:
        x (ndarray): Input tensor with shape (mb x w x w x l)
        k (int): Kernel width for pooling (will be k x k, spatially)
        s: s (int): Stride

    Returns:
        (ndarray): Output of max pooling layer a tensor with shape
            (mb x w_prime x w_prime x l_prime)
               (will be MB x w_prime x w_prime x L)
    """
    return tf.nn.max_pool(x, [1, k, k, 1], [1, s, s, 1], padding="SAME")
