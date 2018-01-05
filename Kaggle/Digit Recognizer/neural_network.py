import os
import tensorflow as tf
from data import read_file
from data import get_batch
from data import FEATURES
from data import OUTPUTS

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
    everything = DataPoint()
    output = DataPoint()

    def __init__(self, training_file, usage_file, split_type="test", cv_percent=0.2, test_percent=0.2):
        # Start by reading in our CSV files.
        training_data, training_data_size = read_file(training_file)
        usage_data, usage_data_size = read_file(usage_file)

        self.everything.indexes, self.everything.xs, self.everything.ys = get_batch(training_data, 0, training_data_size)
        self.everything.length = training_data_size

        if split_type == "cv-test":
            # Get our training data.
            self.training.indexes, self.training.xs, self.training.ys = get_batch(training_data, 0, int(training_data_size * (1 - cv_percent - test_percent)))
            self.training.length = int(training_data_size * (1 - cv_percent - test_percent))

            # Get our cross validation data.
            self.cross_validation.indexes, self.cross_validation.xs, self.cross_validation.ys = get_batch(training_data, int(training_data_size * (1 - cv_percent - test_percent)), int(training_data_size * (1 - test_percent)))
            self.cross_validation.length = int(training_data_size * (1 - test_percent)) - int(training_data_size * (1 - cv_percent - test_percent))

            # Get our testing data.
            self.testing.indexes, self.testing.xs, self.testing.ys = get_batch(training_data, int(training_data_size * (1 - test_percent)), training_data_size)
            self.testing.length = training_data_size - int(training_data_size * (1 - test_percent))

            # Get our output data.
            self.output.indexes, self.output.xs, self.output.ys = get_batch(usage_data, 0, usage_data_size)
            self.output.length = usage_data_size
        elif split_type == "test":
            # Get our training data.
            self.training.indexes, self.training.xs, self.training.ys = get_batch(training_data, 0, int(training_data_size * (1 - test_percent)))
            self.training.length = int(training_data_size * (1 - test_percent))

            # Get our testing data.
            self.testing.indexes, self.testing.xs, self.testing.ys = get_batch(training_data, int(training_data_size * (1 - test_percent)), training_data_size)
            self.testing.length = training_data_size - int(training_data_size * (1 - test_percent))

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


# Setup our weights and biases and try to prevent them from
# ever getting to 0 (dying) as best we can.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W")

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="b")

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

with tf.name_scope("prediction"):
    x = tf.placeholder(tf.float32, [None, FEATURES], name="inputs")
    keep_prob = tf.placeholder(tf.float32, name="kP")
    y_ = tf.placeholder(tf.float32, [None, OUTPUTS], name="actuals")
    # Reshape our image.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # We have 3 layers that we pass everything through before the readout.
    layer1 = create_conv_pool(x_image, 1, 32)
    layer2 = create_conv_pool(layer1, 32, 64)
    layer3 = create_conv_pool(layer2, 64, 128)
    layer4 = create_conv_pool(layer3, 128, 256)
    full_connected_layer = create_fc_layer(layer4, 7 * 7 * 256, 1024)
    y_activation = create_dropout_connected_readout(full_connected_layer, 1024, 10)
    prediction = tf.nn.softmax(y_activation)

    # Get our calculated input (1 if survived, 0 otherwise)
    output = tf.argmax(prediction, 1)

# Now calculate the error and train it.
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_activation))
    tf.summary.scalar("cost", cost)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
# Calculate the accuracy.
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

# Calculate our F1 score.
tp = tf.count_nonzero(tf.argmax(prediction, 1) * tf.argmax(y_, 1))
tn = tf.count_nonzero((tf.argmax(prediction, 1) - 1) * (tf.argmax(y_, 1) - 1))
fp = tf.count_nonzero(tf.argmax(prediction, 1) * (tf.argmax(y_, 1) - 1))
fn = tf.count_nonzero((tf.argmax(prediction, 1) - 1) * tf.argmax(y_, 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
tf.summary.scalar("f1", f1)

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
def get_train_accuracy(sess, data):
    return sess.run(accuracy, feed_dict={x: data.training.xs, keep_prob: 1.0, y_: data.training.ys}) * 100
def get_cv_accuracy(sess, data):
    return sess.run(accuracy, feed_dict={x: data.cross_validation.xs, keep_prob: 1.0, y_: data.cross_validation.ys}) * 100
def get_test_accuracy(sess, data):
    return sess.run(accuracy, feed_dict={x: data.testing.xs, keep_prob: 1.0, y_: data.testing.ys}) * 100
def get_total_accuracy(sess, data):
    return sess.run(accuracy, feed_dict={x: data.everything.xs, keep_prob: 1.0, y_: data.everything.ys}) * 100

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
def get_train_f1_score(sess, data):
    return sess.run(f1, feed_dict={x: data.training.xs, keep_prob: 1.0, y_: data.training.ys})
def get_cv_f1_score(sess, data):
    return sess.run(f1, feed_dict={x: data.cross_validation.xs, keep_prob: 1.0, y_: data.cross_validation.ys})
def get_test_f1_score(sess, data):
    return sess.run(f1, feed_dict={x: data.testing.xs, keep_prob: 1.0, y_: data.testing.ys})
def get_total_f1_score(sess, data):
    return sess.run(f1, feed_dict={x: data.everything.xs, keep_prob: 1.0, y_: data.everything.ys})

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

def save_outputs(sess, data, output_file_name, batch=1):
    # And finally write the results to an output file.
    with open(output_file_name, "w") as out_file:
        out_file.write("ImageId,Label\n")
        for i in range (1, batch + 1):
            data_start = int(((i - 1) / batch) * data.output.length)
            data_end = int((i / batch) * data.output.length)
            results = sess.run(output, feed_dict={x: data.output.xs[data_start:data_end], keep_prob: 1.0, y_: data.output.ys[data_start:data_end]})
            for index, prediction in zip(data.output.indexes[data_start:data_end], results):
                out_file.write("{0},{1}\n".format(index[0], prediction))
            if batch > 1:
                progress = int((i / batch) * 10)
                print('\rSaving results: [{0}{1}] {2}%'.format('#' * progress, ' ' * (10 - progress), round(10 * progress, 2)), end=" ")
        if batch > 1:
            print()
