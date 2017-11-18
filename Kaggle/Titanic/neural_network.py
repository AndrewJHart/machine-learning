from data import read_file
from data import get_batch
from data import FEATURES
from data import OUTPUTS
import tensorflow as tf

# Define how much of our training data we want to split.
CV_PERCENT = 0.2
TEST_PERCENT = 0.2

class DataPoint:
    length = 0
    indexes = []
    xs = []
    ys = []

# Our expected inputs for training, testing, etc.
class NNData:
    training = DataPoint()
    cross_validation = DataPoint()
    testing = DataPoint()
    output = DataPoint()

    def __init__(self, training_file, usage_file, split_training=True):
        # Start by reading in our CSV files.
        training_data, training_data_size = read_file(training_file)
        usage_data, usage_data_size = read_file(usage_file)

        if split_training:
            # Get our training data.
            self.training.indexes, self.training.xs, self.training.ys = get_batch(training_data, 0, int(training_data_size * (1 - CV_PERCENT - TEST_PERCENT)))
            self.training.length = int(training_data_size * (1 - CV_PERCENT - TEST_PERCENT))

            # Get our cross validation data.
            self.cross_validation.indexes, self.cross_validation.xs, self.cross_validation.ys = get_batch(training_data, int(training_data_size * (1 - CV_PERCENT - TEST_PERCENT)), int(training_data_size * (1 - TEST_PERCENT)))
            self.cross_validation.length = int(training_data_size * (1 - TEST_PERCENT)) - int(training_data_size * (1 - CV_PERCENT - TEST_PERCENT))

            # Get our testing data.
            self.testing.indexes, self.testing.xs, self.testing.ys = get_batch(training_data, int(training_data_size * (1 - TEST_PERCENT)), training_data_size)
            self.testing.length = training_data_size - int(training_data_size * (1 - TEST_PERCENT))

            # Get our output data.
            self.output.indexes, self.output.xs, self.output.ys = get_batch(usage_data, 0, usage_data_size)
            self.output.length = usage_data_size
        else:
            # Get our training data.
            self.training.indexes, self.training.xs, self.training.ys = get_batch(training_data, 0, training_data_size)
            self.training.length = training_data_size

            # Get our output data.
            self.output.indexes, self.output.xs, self.output.ys = get_batch(usage_data, 0, usage_data_size)
            self.output.length = usage_data_size


# Setup some helper methods.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W")

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="b")

def create_layer(name, input, input_shape, output_shape, softmax=False):
    # Create our weights and calculate our prediction.
    W = weight_variable([input_shape, output_shape])
    b = bias_variable([output_shape])
    y = tf.matmul(input, W) + b
    if softmax:
        y = tf.nn.softmax(y)
    # Give some summaries for the outputs.
    tf.summary.histogram("weights_" + name, W)
    tf.summary.histogram("biases_" + name, b)
    tf.summary.histogram("y_" + name, y)
    return y

with tf.name_scope("prediction"):
    x = tf.placeholder(tf.float32, [None, FEATURES], name="inputs")
    y_ = tf.placeholder(tf.float32, [None, OUTPUTS], name="actuals")

    # 3 layers (1 input, 1 hidden, 1 output).
    y_input = create_layer("input", x, FEATURES, FEATURES, softmax=True)
    y_hidden = create_layer("hidden", y_input, FEATURES, OUTPUTS, softmax=True)
    y_activation = create_layer("activation", y_hidden, OUTPUTS, OUTPUTS, softmax=True)

    # Get our calculated input (1 if survived, 0 otherwise)
    output = tf.argmax(y_activation, 1)

# Now calculate the error and train it.
with tf.name_scope("cost"):
    cost = -(tf.reduce_sum(y_ * tf.log(y_activation) + (1 - y_) * tf.log(1 - y_activation)))
    tf.summary.scalar("cost", cost)
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
# Calculate the accuracy finally.
correct_prediction = tf.equal(tf.argmax(y_activation, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

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
        sess.run(train_step, feed_dict={x: data.training.xs[data_start:data_end], y_: data.training.ys[data_start:data_end]})
        training_iteration += 1

def train_summary(sess, data, merged_summary, writer, batch=1):
    global training_iteration
    for i in range (1, batch + 1):
        data_start = int(((i - 1) / batch) * data.training.length)
        data_end = int((i / batch) * data.training.length)
        s, t = sess.run([merged_summary, train_step], feed_dict={x: data.training.xs[data_start:data_end], y_: data.training.ys[data_start:data_end]})
        writer.add_summary(s, training_iteration)
        training_iteration += 1

# Accuracy methods.
def get_train_accuracy(sess, data):
    return sess.run(accuracy, feed_dict={x: data.training.xs, y_: data.training.ys}) * 100
def get_cv_accuracy(sess, data):
    return sess.run(accuracy, feed_dict={x: data.cross_validation.xs, y_: data.cross_validation.ys}) * 100
def get_test_accuracy(sess, data):
    return sess.run(accuracy, feed_dict={x: data.testing.xs, y_: data.testing.ys}) * 100

def save_outputs(sess, data, output_file_name):
    # And finally write the results to an output file.
    with open(output_file_name, "w") as out_file:
        out_file.write("PassengerId,Survived\n")
        results = sess.run(output, feed_dict={x: data.output.xs, y_: data.output.ys})
        for index, prediction in zip(data.output.indexes, results):
            out_file.write("{0},{1}\n".format(index[0], prediction))
