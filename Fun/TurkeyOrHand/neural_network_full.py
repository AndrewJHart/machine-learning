import tensorflow as tf
import file_reader as fr

CATEGORIES = ['live_turkey', 'cooked_turkey', 'hand_turkey', 'hand_palm']

# Create our placeholders and load our data. keep_prob is the probability we
# will keep a weight during training.
filename = tf.placeholder(tf.string, name="filename")
keep_prob = tf.placeholder(tf.float32, name="kP")

x_image = [fr.read_image(filename)]
y = [[0, 0, 0, 1]]

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
layer1 = create_conv_pool(x_image, 3, 32 * 3)
layer2 = create_conv_pool(layer1, 32 * 3, 64 * 3)
full_connected_layer = create_fc_layer(layer2, 7 * 7 * 64 * 3, 1024 * 3)
prediction = tf.nn.softmax(create_dropout_connected_readout(full_connected_layer, 1024 * 3, 4))
output = tf.argmax(prediction, 1)

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

def get_prediction(sess, testImageName):
    pred, out = sess.run([prediction, output], feed_dict={filename: testImageName, keep_prob: 0.5})
    return CATEGORIES[out[0]], pred
