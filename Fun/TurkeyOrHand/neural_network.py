import os
import tensorflow as tf
import file_reader as fr

CATEGORIES = ['live_turkey', 'cooked_turkey', 'hand_turkey', 'hand_palm']

# Create our placeholders and load our data. keep_prob is the probability we
# will keep a weight during training.
x = tf.placeholder(tf.float32, shape=[None, 784 * 3], name="images")
y = tf.placeholder(tf.float32, shape=[None, 4], name="y")
keep_prob = tf.placeholder(tf.float32, name="kP")

x_image = tf.reshape(x, [-1, 28, 28, 3])

# Setup some helper methods.
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

# We have 3 layers that we pass everything through before the readout.
layer1 = create_conv_pool(x_image, 3, 32)
layer2 = create_conv_pool(layer1, 32, 64)
full_connected_layer = create_fc_layer(layer2, 7 * 7 * 64, 1024)
prediction = create_dropout_connected_readout(full_connected_layer, 1024, 4)
output = tf.argmax(prediction, 1)

# Now we train the model and evaluate its accuracy.
with tf.name_scope("cost_calc"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('cost', cost)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Create an embedding to display our fully connected layer visually.
embedding = tf.Variable(tf.zeros([1024, 1024]), name="test_embedding")
# assignment = embedding.assign(full_connected_layer)
saver = tf.train.Saver()

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
    sess.run(train_step, feed_dict={x: data.images, y: data.labels, keep_prob: 0.5})
    training_iteration += 1

def train_summary(sess, data, merged_summary, writer):
    global training_iteration
    s, t = sess.run([merged_summary, train_step], feed_dict={x: data.images, y: data.labels, keep_prob: 0.5})
    writer.add_summary(s, training_iteration)
    training_iteration += 1

# Accuracy methods.
def get_accuracy(sess, data):
    return sess.run(accuracy, feed_dict={x: data.images, y: data.labels, keep_prob: 0.5}) * 100

def get_prediction(sess, image):
    image_data = fr.read_image(image)
    out, pred = sess.run([output, prediction], feed_dict={x: [image_data], y: [[0, 0, 0, 0]], keep_prob: 0.5})
    name = CATEGORIES[out[0]]
    return name, pred

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