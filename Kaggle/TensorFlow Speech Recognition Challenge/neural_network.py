import os
import tensorflow as tf
import numpy as np
from data import get_files
from data import FEATURES
from data import OUTPUTS
from data import RESULT_MAP

class DataPoint:
    length = 0
    filenames = []
    xs = []
    ys = []

# Our expected inputs for training, testing, etc.
class NNData:
    training = DataPoint()
    cross_validation = DataPoint()
    testing = DataPoint()
    everything = DataPoint()
    output = DataPoint()

    def __init__(self, load_training_data=True, partial=False, split_type="test", cv_percent=0.2, test_percent=0.2):
        # Start by reading in our CSV files.
        self.everything.xs, self.everything.ys, self.everything.filenames = get_files(partial, training_data=load_training_data)
        self.everything.length = len(self.everything.xs)

        if split_type == "cv-test":
            # Get our training data.
            self.training.xs = self.everything.xs[0:int(self.everything.length * (1 - cv_percent - test_percent))]
            self.training.ys = self.everything.ys[0:int(self.everything.length * (1 - cv_percent - test_percent))]
            self.training.filenames = self.everything.filenames[0:int(self.everything.length * (1 - cv_percent - test_percent))]
            self.training.length = int(self.everything.length * (1 - cv_percent - test_percent))

            # Get our cross validation data.
            self.cross_validation.xs = self.everything.xs[int(self.everything.length * (1 - cv_percent - test_percent)):int(self.everything.length * (1 - test_percent))]
            self.cross_validation.ys = self.everything.ys[int(self.everything.length * (1 - cv_percent - test_percent)):int(self.everything.length * (1 - test_percent))]
            self.cross_validation.filenames = self.everything.filenames[int(self.everything.length * (1 - cv_percent - test_percent)):int(self.everything.length * (1 - test_percent))]
            self.cross_validation.length = int(self.everything.length * (1 - test_percent)) - int(self.everything.length * (1 - cv_percent - test_percent))

            # Get our testing data.
            self.testing.xs = self.everything.xs[int(self.everything.length * (1 - test_percent)):self.everything.length]
            self.testing.ys = self.everything.ys[int(self.everything.length * (1 - test_percent)):self.everything.length]
            self.testing.filenames = self.everything.filenames[int(self.everything.length * (1 - test_percent)):self.everything.length]
            self.testing.length = self.everything.length - int(self.everything.length * (1 - test_percent))
        elif split_type == "test":
            # Get our training data.
            self.training.xs = self.everything.xs[0:int(self.everything.length * (1 - test_percent))]
            self.training.ys = self.everything.ys[0:int(self.everything.length * (1 - test_percent))]
            self.training.filenames = self.everything.filenames[0:int(self.everything.length * (1 - test_percent))]
            self.training.length = int(self.everything.length * (1 - test_percent))

            # Get our testing data.
            self.testing.xs = self.everything.xs[int(self.everything.length * (1 - test_percent)):self.everything.length]
            self.testing.ys = self.everything.ys[int(self.everything.length * (1 - test_percent)):self.everything.length]
            self.testing.filenames = self.everything.filenames[int(self.everything.length * (1 - test_percent)):self.everything.length]
            self.testing.length = self.everything.length - int(self.everything.length * (1 - test_percent))
        else:
            # Get our training data.
            self.training.xs = self.everything.xs[0:self.everything.length]
            self.training.ys = self.everything.ys[0:self.everything.length]
            self.training.filenames = self.everything.filenames[0:self.everything.length]
            self.training.length = self.everything.length

# Setup our weights and biases and try to prevent them from
# ever getting to 0 (dying) as best we can.
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W" + name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="b" + name)

# Setup our convolution (with strides of 1).
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Setup our pooling options (2x2 matrix for pooling)
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

# Creates a convolutional pooling network.
def create_conv_pool(input, input_size, output_size, name="conv"):
    with tf.name_scope(name):
        weights = weight_variable([5, 5, input_size, output_size])
        biases = bias_variable([output_size])
        activation = tf.nn.relu(conv2d(input, weights) + biases)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", activation)
        return max_pool_2x2(activation)

# Creates our fully connected layer.
def create_fc_layer(input, input_size, output_size, name="fc"):
    with tf.name_scope(name):
        weights = weight_variable([input_size, output_size])
        biases = bias_variable([output_size])
        flat_input = tf.reshape(input, [-1, input_size])
        return tf.nn.relu(tf.matmul(flat_input, weights) + biases)

# Creates our dropout layer which is used for our prediction.
def create_dropout_connected_readout(input, input_size, output_size, name="readout"):
    with tf.name_scope(name):
        weights = weight_variable([input_size, output_size])
        biases = bias_variable([output_size])
        weight_dropout = tf.nn.dropout(weights, keep_prob) * keep_prob
        return tf.matmul(input, weight_dropout) + biases

def create_layer(name, input, input_shape, output_shape, activation_function=None):
    # Create our weights and calculate our prediction.
    W = weight_variable([input_shape, output_shape], name)
    b = bias_variable([output_shape], name)
    y = tf.matmul(input, W) + b
    if activation_function is "softmax":
        y = tf.nn.softmax(y)
    if activation_function is "relu":
        y = tf.nn.relu(y)
    # Give some summaries for the outputs.
    tf.summary.histogram("weights_" + name, W)
    tf.summary.histogram("biases_" + name, b)
    tf.summary.histogram("y_" + name, y)
    return W, y

with tf.name_scope("prediction"):
    x = tf.placeholder(tf.float32, [None, FEATURES], name="inputs")
    keep_prob = tf.placeholder(tf.float32, name="kP")
    y_ = tf.placeholder(tf.float32, [None, OUTPUTS], name="actuals")

    W_input, y_input = create_layer("input", x, FEATURES, 100, activation_function="softmax")
    W_hidden, y_hidden = create_layer("hidden", y_input, 100, 100, activation_function=None)
    W_activation, y_activation = create_layer("activation", y_hidden, 100, OUTPUTS, activation_function=None)
    prediction = tf.nn.softmax(y_activation)

    # Get our calculated input (1 if survived, 0 otherwise)
    output = tf.argmax(prediction, 1)

# Now calculate the error and train it.
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_activation))
    tf.summary.scalar("cost", cost)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

# Calculate the list of correct predictions.
correct_prediction = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1)), tf.float32)

# We want calculate the accuracy from an input, that way we can batch everything.
prediction_results = tf.placeholder_with_default(correct_prediction, [None], name="prediction_results")
accuracy = tf.reduce_mean(prediction_results)
tf.summary.scalar("accuracy", accuracy)

# Calculate our F1 score. Need to learn how to do with multiclass.
# tp = tf.count_nonzero(tf.argmax(prediction, 1) * tf.argmax(y_, 1))
# tn = tf.count_nonzero((tf.argmax(prediction, 1) - 1) * (tf.argmax(y_, 1) - 1))
# fp = tf.count_nonzero(tf.argmax(prediction, 1) * (tf.argmax(y_, 1) - 1))
# fn = tf.count_nonzero((tf.argmax(prediction, 1) - 1) * tf.argmax(y_, 1))
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1 = 2 * precision * recall / (precision + recall)
# tf.summary.scalar("f1", f1)

saver = tf.train.Saver(tf.global_variables())

#############################################################
#############################################################
#############################################################
# Helper methods for utilizing the neural network.  #########
#############################################################
#############################################################
#############################################################

def setup(log_dir):
    # Setup our session.
    sess = tf.InteractiveSession()
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir)
    writer.add_graph(sess.graph)
    tf.global_variables_initializer().run()
    return sess, merged_summary, writer

# Keep track of our training iterations.
training_iteration = 0

def train(sess, data, batch=1):
    global training_iteration
    for i in range (1, batch + 1):
        data_start = int(((i - 1) / batch) * data.training.length)
        data_end = int((i / batch) * data.training.length)
        sess.run(train_step, feed_dict={x: data.training.xs[data_start:data_end], keep_prob: 0.5, y_: data.training.ys[data_start:data_end]})
        training_iteration += 1
        if batch > 1:
            progress = int((i / (batch + 1)) * 10)
            print('\rBatched Training: [{0}{1}] {2}%'.format('#' * progress, ' ' * (10 - progress), round(10 * progress, 2)), end=" ")
    if batch > 1:
        print()

def train_summary(sess, data, merged_summary, writer, batch=1):
    global training_iteration
    for i in range (1, batch + 1):
        data_start = int(((i - 1) / batch) * data.training.length)
        data_end = int((i / batch) * data.training.length)
        s, t = sess.run([merged_summary, train_step], feed_dict={x: data.training.xs[data_start:data_end], keep_prob: 0.5, y_: data.training.ys[data_start:data_end]})
        writer.add_summary(s, training_iteration)
        training_iteration += 1
        if batch > 1:
            progress = int((i / batch) * 10)
            print('\rBatched Training: [{0}{1}] {2}%'.format('#' * progress, ' ' * (10 - progress), round(10 * progress, 2)), end=" ")
    if batch > 1:
        print()

# Accuracy methods.
def get_train_accuracy(sess, data, batch=1):
    total_predictions = []
    for i in range (1, batch + 1):
        data_start = int(((i - 1) / batch) * data.training.length)
        data_end = int((i / batch) * data.training.length)
        predictions = sess.run(correct_prediction, feed_dict={x: data.training.xs[data_start:data_end], keep_prob: 1.0, y_: data.training.ys[data_start:data_end]})
        total_predictions = np.concatenate((total_predictions, predictions), axis=0)
    return sess.run(accuracy, feed_dict={prediction_results: total_predictions}) * 100

def get_cv_accuracy(sess, data, batch=1):
    total_predictions = []
    for i in range (1, batch + 1):
        data_start = int(((i - 1) / batch) * data.cross_validation.length)
        data_end = int((i / batch) * data.cross_validation.length)
        predictions = sess.run(correct_prediction, feed_dict={x: data.cross_validation.xs[data_start:data_end], keep_prob: 1.0, y_: data.cross_validation.ys[data_start:data_end]})
        total_predictions += predictions
    return sess.run(accuracy, feed_dict={prediction_results: total_predictions}) * 100

def get_test_accuracy(sess, data, batch=1):
    total_predictions = []
    for i in range (1, batch + 1):
        data_start = int(((i - 1) / batch) * data.testing.length)
        data_end = int((i / batch) * data.testing.length)
        predictions = sess.run(correct_prediction, feed_dict={x: data.testing.xs[data_start:data_end], keep_prob: 1.0, y_: data.testing.ys[data_start:data_end]})
        total_predictions += predictions
    return sess.run(accuracy, feed_dict={prediction_results: total_predictions}) * 100

def get_total_accuracy(sess, data, batch=1):
    total_predictions = []
    for i in range (1, batch + 1):
        data_start = int(((i - 1) / batch) * data.everything.length)
        data_end = int((i / batch) * data.everything.length)
        predictions = sess.run(correct_prediction, feed_dict={x: data.everything.xs[data_start:data_end], keep_prob: 1.0, y_: data.everything.ys[data_start:data_end]})
        total_predictions += predictions
    return sess.run(accuracy, feed_dict={prediction_results: total_predictions}) * 100

# Cost methods.
def get_train_cost(sess, data):
    return sess.run(cost, feed_dict={x: data.training.xs, keep_prob: 1.0, y_: data.training.ys})
def get_cv_cost(sess, data):
    return sess.run(cost, feed_dict={x: data.cross_validation.xs, keep_prob: 1.0, y_: data.cross_validation.ys})
def get_test_cost(sess, data):
    return sess.run(cost, feed_dict={x: data.testing.xs, keep_prob: 1.0, y_: data.testing.ys})
def get_total_cost(sess, data):
    return sess.run(cost, feed_dict={x: data.everything.xs, keep_prob: 1.0, y_: data.everything.ys})

# F1 Score methods.
# def get_train_f1_score(sess, data):
#     return sess.run(f1, feed_dict={x: data.training.xs, keep_prob: 1.0, y_: data.training.ys})
# def get_cv_f1_score(sess, data):
#     return sess.run(f1, feed_dict={x: data.cross_validation.xs, keep_prob: 1.0, y_: data.cross_validation.ys})
# def get_test_f1_score(sess, data):
#     return sess.run(f1, feed_dict={x: data.testing.xs, keep_prob: 1.0, y_: data.testing.ys})
# def get_total_f1_score(sess, data):
#     return sess.run(f1, feed_dict={x: data.everything.xs, keep_prob: 1.0, y_: data.everything.ys})

def reset():
    tf.global_variables_initializer().run()

def load_model(sess, model_name, directory="model"):
    if os.path.exists(directory):
        saver.restore(sess, directory + "/" + model_name);
    else:
        print("Error loading model!")
        exit(-1)

def save_model(sess, model_name, directory="model"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    saver.save(sess, directory + "/" + model_name); 

def save_outputs(sess, data, output_file_name, include_actual=True, batch=1):
    # And finally write the results to an output file.
    with open(output_file_name, "w") as out_file:
        out_file.write("fname,label{0}\n".format(",actual" if include_actual else ""))
        for i in range (1, batch + 1):
            data_start = int(((i - 1) / batch) * data.everything.length)
            data_end = int((i / batch) * data.everything.length)
            results = sess.run(output, feed_dict={x: data.everything.xs[data_start:data_end], keep_prob: 1.0, y_: data.everything.ys[data_start:data_end]})
            for filename, prediction, actual in zip(data.everything.filenames[data_start:data_end], results, data.everything.ys[data_start:data_end]):
                out_file.write("{0},{1}{2}\n".format(filename, RESULT_MAP[prediction], ("," + RESULT_MAP[actual.index(1)]) if include_actual else ""))
            if batch > 1:
                progress = int((i / batch) * 10)
                print('\rSaving results: [{0}{1}] {2}%'.format('#' * progress, ' ' * (10 - progress), round(10 * progress, 2)), end=" ")
        if batch > 1:
            print()
