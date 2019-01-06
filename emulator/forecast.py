import os
import tensorflow as tf
import numpy as np


def load_pretrained_graph():
    """
    Loads the pretrained 18-layer Residual Network from protobuf file.

    Returns:
        (tf.Graph) TensorFlow graph object.
    """
    pb_path = "%s/pretrained_models/18-layer_residual_network.pb" % \
              os.path.dirname(__file__)

    # Retrieve unserialized graph definition
    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import the graph_def into a new TensorFlow Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


def gen_forecasts(input_tensor, num_forecasts, model_path=None):
    """
    Generates 'num_forecasts' precipitation forecasts conditioned on the
    precipitation maps in the ndarray 'input_tensor'.

    Args:
        input_tensor (ndarray): Precipitation maps (window_size x 64 x 128)
        num_forecasts (int): Number of forecasts to generate
        model_path (str): Path to TensorFlow checkpoint files. Uses
                          pretrained 18-layer Residual Network if files are
                          not provided

    Returns:
        (ndarray) Sequential list of precipitation forecasts.
    """
    forecasts = []
    window_size = input_tensor.shape[0]

    # Create TensorFlow Session
    if model_path is None:
        print("No checkpoint files provided. Defaulting to pretrained 18-layer "
              "Residual Network...")
        sess = tf.Session(graph=load_pretrained_graph())
    else:
        sess = tf.Session()

        print("Restoring model from TensorFlow checkpoint files...")
        saver = tf.train.import_meta_graph(model_path + "model.meta")
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

    # Contains the previous [1,...,(window_size-1)] precipitation maps
    prev = input_tensor[0:window_size - 1]

    # Contains the most recent precipitation field
    forecast = input_tensor[window_size - 1]

    for i in range(num_forecasts):
        # Reshape because our graph outputs a tensor with shape (64,128).
        forecast = np.reshape(forecast, (1, 64, 128))

        # Append 'forecast' (our model's output) to our input tensor.
        x = np.concatenate((prev, forecast), axis=0)
        x = np.reshape(x, (1, 64, 128, window_size))

        # Feed input tensor into model, producing a precipitation forecast
        forecast = sess.run(fetches=["z:0"], feed_dict={"x:0": x})
        forecasts.append(np.squeeze(forecast))

        # Shift the window by slicing off the earliest timestep.
        x = np.reshape(x, (window_size, 64, 128, 1))
        prev = x[1:]
        prev = np.squeeze(prev)

    sess.close()

    return np.asarray(forecasts)
