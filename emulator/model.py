import datetime
import logging
import os
import sys
from functools import partial
from subprocess import call
import numpy as np
import tensorflow as tf
import emulator.data as data
import emulator.error as error
import emulator.graph as graph
from emulator.normalization import Normalizer


# Static decay functions
def linear(decay_const, epoch):
    """
    Applies linear decay as a function of the number of training epochs.

    Args:
        decay_const (str): Constant use in linear decay function
        epoch (int): Current training epoch

    Returns:
        (float) Updated scheduled sampling rate
    """
    return np.maximum(0, 1 + (decay_const * epoch))


def exp(decay_const, epoch):
    """
        Applies exponential decay as a function of the number of training
        epochs.

        Args:
            decay_const (str): Constant use in exponential decay function
            epoch (int): Current training epoch

        Returns:
            (float) Updated scheduled sampling rate
        """
    return decay_const ** epoch


def build_decay_func(decay_func, epochs):
    """
    Higher order function that builds and returns the specified decay function.

    Args:
        decay_func (str): Specifies decay function in the set {"linear", "exp"}
        epochs (int): Number of training epochs

    Returns:
        (func) Partial function used to decay the scheduled sampling probability
    """
    if decay_func == "linear":
        decay_const = -1.0 / float(epochs)
        return partial(linear, decay_const)
    elif decay_func == "exp":
        decay_const = np.power(0.01, 1.0/float(epochs))
        return partial(exp, decay_const)


class ResNet:
    def __init__(self, architecture, hyperparameters, pr_field, train_pct=0.7,
                 save_model=True):
        """
        Residual Network constructor. Builds computational graph and stores
        training data and hyperparameters.

        Args:
            architecture (dict): Contains DNN architecture definition
            hyperparameters (dict): Contains hyperparameters
            pr_field (ndarray): Contains all precipitation maps
            train_pct (float): Percent of data to be used for training
            save_model (bool): If True, save model checkpoints during training
        """
        # Load area_weights from csv.
        self.area_weights = data.load_area_weights()

        # Split data set
        self.train, self.dev, self.test = \
            data.train_dev_test_split(pr_field, train_pct=train_pct)

        # Data preprocessing
        self.norm = Normalizer()
        normalized_data = self.norm.transform(pr_field, len(self.train), copy=True)
        self.normalized_train, self.normalized_dev, self.normalized_test = \
            data.train_dev_test_split(normalized_data, train_pct=train_pct)

        # Hyperparameters
        self.architecture = architecture                                    # Model definition
        self.epochs = int(hyperparameters["epochs"])                        # Number of epochs to train on
        self.lr = float(hyperparameters["lr"])                              # Learning Rate
        self.window_size = int(hyperparameters["window"])                   # Window size
        self.tn_stddev = float(hyperparameters["truncated_normal_stddev"])  # Standard deviation value for truncated normal initializer
        self.sched = bool(hyperparameters["use_scheduled_sampling"])        # Train with scheduled sampling when True
        self.decay_func = build_decay_func(str(hyperparameters["decay_func"]), self.epochs)  # Sampling decay function

        # Optional patience threshold used for early stopping
        # (when training without scheduled sampling)
        self.patience = 10 if "patience" not in hyperparameters \
            else int(hyperparameters["patience"])

        assert self.window_size < len(self.dev), \
            "Window size must be less than the length of the validation set."

        # Create subdirectory to store model weights and training log.
        self.model_dir = None
        if save_model:
            self.create_model_dir()
        else:
            logging.basicConfig(filename='training.log',
                                level=logging.DEBUG, filemode='a',
                                format='%(asctime)s - %(levelname)s: %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p')

        # Build computational graph for ResNet.
        self.build_graph(bool(hyperparameters["use_area_weighted_obj"]))

    def create_model_dir(self):
        """
        Create output directory for checkpoint files and training log.
        """
        if not os.path.isdir("./saved_models"):
            call("mkdir saved_models", shell=True)
            print('Creating directory \"saved_models\" to store model '
                  'checkpoints)')

        # Create unique output directory
        self.model_dir = "saved_models/%s" % datetime.datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S")
        call("mkdir %s" % self.model_dir, shell=True)

        # Set up log file to record general information about program operation
        logging.basicConfig(filename='%s/training.log' % self.model_dir,
                            level=logging.DEBUG, filemode='a',
                            format='%(asctime)s - %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    def build_graph(self, use_area_weighted_obj):
        """
        Constructs TensorFlow computational graph.

        Args:
            use_area_weighted_obj (bool): If true, train using an
                area-weighted objective function
        """
        tf.reset_default_graph()

        # Placeholders
        area_weights = tf.placeholder(dtype=tf.float32, shape=None,
                                      name="area_weights")
        y_true = tf.placeholder(dtype=tf.float32, shape=None, name="y_true")
        x = tf.placeholder(dtype=tf.float32,
                           shape=(None, 64, 128, self.window_size), name="x")

        # Construct graph according to self.architecture
        unit_id = 1
        prev_a = x
        prev_lprime = self.window_size
        for layer in self.architecture["layers"]:
            if "shortcut" in layer:

                assert "layers" in layer, \
                    "Expected shortcut connection to have nested layers."
                assert "projection" in layer, \
                    "Expected shortcut connection to specify use of projection."

                # Build a residual unit
                with tf.variable_scope("shortcut_" + str(unit_id)):
                    prev_a, prev_lprime = graph.residual_unit(
                        prev_a=prev_a, prev_lprime=prev_lprime,
                        layers=layer["layers"], tn_stddev=self.tn_stddev,
                        projection=bool(layer["projection"]))
            else:
                # Build a standard convolutional layer
                prev_a = graph.conv2d(prev_a=prev_a, k=layer["kernel_width"],
                                   l=prev_lprime, l_prime=layer["filters"],
                                   s=layer["stride"], p=layer["padding"],
                                   act=layer["activation"])

            unit_id += 1

        z = tf.squeeze(prev_a, name="z")

        # define loss
        area_weighted_obj = tf.reduce_mean(np.square(np.multiply((z - y_true),
                                                                 area_weights)),
                                           name="area_weighted_obj")
        obj = tf.reduce_mean(tf.squared_difference(z, y_true), name="obj")

        # Used when evaluating on dev
        _ = tf.squared_difference(z, y_true, name="squared_diff")

        # Define our optimizer
        if use_area_weighted_obj:
            _ = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(area_weighted_obj, name="train_step")
        else:
            _ = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(obj, name="train_step")

    def fit(self):
        """
        Trains DNN until convergence criteria is met.

        Returns:
            (float): Mean Squared Error over all possible 1-timestep forecasts
            in the dev set.
        """
        saver = tf.train.Saver()

        # Training Loop
        with tf.Session() as sess:
            print('Starting training session...')
            logging.info("Starting training session...")

            sess.run(tf.global_variables_initializer())

            # Used for early stopping
            best_dev_err = float('inf')
            bad_count = 0

            train_size = len(self.normalized_train) - self.window_size
            print('Length of Training Set: %d' % train_size)

            epsilon = 1.0
            for epoch in range(self.epochs):

                # Initializing train_z as the nth map in window, and prev as
                # maps 1 to n-1 stacked.
                train_z = self.normalized_train[self.window_size - 1]
                prev = self.normalized_train[:self.window_size - 1]

                # Loop over all inputs
                for i in range(train_size):
                    # Reshape because our graph outputs z as a tensor with
                    # shape=(64, 128).
                    train_z = np.reshape(train_z, (1, 64, 128))

                    # Append train_z (either a ground truth map or our
                    # model's output) to our input tensor.
                    train_x = np.concatenate((prev, train_z), axis=0)
                    train_x = np.reshape(train_x, (1, 64, 128, self.window_size))

                    train_y = self.normalized_train[i + self.window_size]
                    train_y = np.reshape(train_y, (1, 64, 128))

                    _, _, _, train_z = sess.run(fetches=["train_step", "obj:0",
                                                         "area_weighted_obj:0",
                                                         "z:0"],
                                                feed_dict={"x:0": train_x,
                                                           "y_true:0": train_y,
                                                           "area_weights:0":
                                                               self.area_weights})

                    # This is where we use scheduled sampling to set up our
                    # next input. Flip a biased coin, take ground truth with a
                    # probability of 'epsilon', take model output with
                    # probability (1 - epsilon).
                    if not self.sched or np.random.binomial(1, epsilon, 1)[0]:
                        train_z = train_y

                    train_x = np.reshape(train_x, (self.window_size, 64, 128, 1))

                    # Shift the window by slicing off the earliest time step.
                    prev = train_x[1:]
                    prev = np.squeeze(prev)

                logging.info("Epoch %d done. Evaluating on dev set..." % epoch)

                # Evaluate performance on the dev set.
                dev_size = len(self.normalized_dev) - self.window_size
                squared_diffs = []
                dev_predictions = []
                for i in range(dev_size):
                    dev_x = self.normalized_dev[i:i + self.window_size]
                    dev_x = np.reshape(dev_x, (1, 64, 128, self.window_size))

                    dev_y = self.normalized_dev[i + self.window_size]
                    dev_y = np.reshape(dev_y, (1, 64, 128))

                    [squared_diff, dev_z, my_weighted_dev_err] = \
                        sess.run(fetches=["squared_diff:0", "z:0",
                                          "area_weighted_obj:0"],
                                 feed_dict={"x:0": dev_x, "y_true:0": dev_y,
                                            "area_weights:0": self.area_weights})

                    squared_diffs.append(squared_diff)
                    dev_predictions.append(dev_z)

                dev_mse = np.mean(squared_diffs)

                # Decay epsilon value is training with scheduled sampling
                if self.sched:
                    epsilon = self.decay_func(epoch)
                    logging.info("Decay applied. epsilon=%.2f" % (epsilon))

                bad_count += 1
                if dev_mse < best_dev_err:
                    best_dev_err = dev_mse
                    bad_count = 0

                    # Save model parameters
                    if self.model_dir:
                        saver.save(sess, "%s/model" % self.model_dir)
                        logging.info("Saved the current best model")

                # Employ early stopping if we are NOT doing scheduled sampling.
                if not self.sched and bad_count >= self.patience:
                    print('Converged due to early stopping...')
                    logging.info('Converged due to early stopping...')
                    break

                # Computing performance metrics on normalized data
                print("Epoch %d: dev_mse=%.10f dev_area-weighted_mse=%.10f "
                      "bad_count=%d" %
                      (epoch, dev_mse, my_weighted_dev_err, bad_count))
                logging.info("Epoch %d: dev_mse=%.10f dev_"
                             "area-weighted_mse=%.10f bad_count=%d" %
                             (epoch, dev_mse, my_weighted_dev_err, bad_count))

                # If loss is NaN, assume we diverged and return an extremely
                # large loss.
                if np.isnan(dev_mse) or np.isinf(dev_mse):
                    logging.info("Model has diverged. Ending training session.")
                    return sys.float_info.max

            # Evaluate performance on denormalized data.
            dev_predictions = self.norm.inverse_transform(dev_predictions)
            dev_mse = error.compute_mse(self.dev[self.window_size:],
                                        dev_predictions)
            print("After denormalization:\ndev_mse=%.10f, dev_rmse=%.10f" %
                  (dev_mse, np.sqrt(dev_mse)))
            logging.info("After denormalization:\ndev_mse=%.10f, dev_rmse=%.10f" %
                         (dev_mse, np.sqrt(dev_mse)))

        logging.info("Ending training session.")

        return dev_mse
