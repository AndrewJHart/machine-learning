from data import read_file
from data import get_batch
from data import FEATURES
from data import OUTPUTS
import tensorflow as tf

class DataPoint:
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
            self.training.indexes, self.training.xs, self.training.ys = get_batch(training_data, 0, int(training_data_size * 0.6))
            self.cross_validation.indexes, self.cross_validation.xs, self.cross_validation.ys = get_batch(training_data, int(training_data_size * 0.6), int(training_data_size * 0.8))
            self.testing.indexes, self.testing.xs, self.testing.ys = get_batch(training_data, int(training_data_size * 0.8), training_data_size)
            self.output.indexes, self.output.xs, self.output.ys = get_batch(usage_data, 0, usage_data_size)
        else:
            self.training.indexes, self.training.xs, self.training.ys = get_batch(training_data, 0, training_data_size)
            self.output.indexes, self.output.xs, self.output.ys = get_batch(usage_data, 0, usage_data_size)


# Setup some helper methods.
def weight_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name="W")

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="b")

#############################################################
#############################################################
################## Setup TensorFlow #########################
#############################################################
#############################################################
# Next lets go ahead and calculate what our results are.
with tf.name_scope("prediction"):
    x = tf.placeholder(tf.float32, [None, FEATURES], name="inputs")
    y_ = tf.placeholder(tf.float32, [None, OUTPUTS], name="actuals")
    W = weight_variable([FEATURES, OUTPUTS])
    b = bias_variable([OUTPUTS])
    y_mid = tf.matmul(x, W) + b
    prediction = tf.nn.softmax(y_mid)
    output = tf.argmax(prediction, 1)
    # Give some summaries for the outputs.
    tf.summary.histogram("weights", W)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("prediction", prediction)

# Now calculate the error and train it.
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))
    tf.summary.scalar("cost", cost)
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
# Calculate the accuracy finally.
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

def setup(log_dir):
    # Setup our session.
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir)
    writer.add_graph(sess.graph)
    return sess, merged_summary, writer

training_iteration = 0

def train(sess, data):
    global training_iteration
    sess.run(train_step, feed_dict={x: data.training.xs, y_: data.training.ys})
    training_iteration += 1

def train_summary(sess, data, merged_summary, writer):
    global training_iteration
    s, t = sess.run([merged_summary, train_step], feed_dict={x: data.training.xs, y_: data.training.ys})
    writer.add_summary(s, training_iteration)
    training_iteration += 1

# Accuracy methods.
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
