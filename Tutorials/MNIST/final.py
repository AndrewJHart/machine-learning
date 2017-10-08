import os
import tensorflow as tf

mnist = tf.contrib.learn.datasets.mnist.read_data_sets("MNIST_data/", one_hot=True)
LOG_DIR = "mnist_visualization"

# Create our placeholders. keep_prob is the probability we
# will keep a weight during training.
x = tf.placeholder(tf.float32, [None, 784], name="inputs")
keep_prob = tf.placeholder(tf.float32, name="kP")
y = tf.placeholder(tf.float32, [None, 10], name="labels")

# Reshape our image data that way we can display it
# with tensorboard.
x_image = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', x_image, 3)
# Create a global variable for our embedding.
global embedding_input

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

####################################################################################################
####################################################################################################
#########################                                 ##########################################
######################### The actual setup for tensorflow ##########################################
#########################                                 ##########################################
####################################################################################################
####################################################################################################
# We have 3 layers that we pass everything through before the readout.
layer1 = create_conv_pool(x_image, 1, 32)
layer2 = create_conv_pool(layer1, 32, 64)
full_connected_layer = create_fc_layer(layer2, 7 * 7 * 64, 1024)
prediction = create_dropout_connected_readout(full_connected_layer, 1024, 10)

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
assignment = embedding.assign(full_connected_layer)
saver = tf.train.Saver()

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

def setup():
    # Create our TensorFlow session and initialize all of our variables.
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    # Setup our graph before we initialize our variables.
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR)
    writer.add_graph(sess.graph)
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = os.path.join(os.getcwd(), "sprite_1024.png")
    embedding_config.metadata_path = os.path.join(os.getcwd(), "labels_1024.tsv")
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    return sess, merged_summary, writer

def train(sess, merged_summary, writer):
    # Start by training the model 20,000 rounds grabbing 100 training
    # examples during each round.
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        if i % 5 == 0:
            s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            writer.add_summary(s, i)
        if i % 500 == 0:
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024], keep_prob: 0.5})
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), i)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})

def print_accuracy(sess):
    with tf.name_scope("accuracy"):
        # Finally, calculate our accuracy and print it out.
        print("======================")
        print("======================")
        print("======================")
        print("======================")
        print("Caculating accuracy...")
        print("======================")
        print("======================")
        print("Accuracy: {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}) * 100))
        print("======================")
        print("======================")

sess, merged_summary, writer = setup()
train(sess, merged_summary, writer)
print_accuracy(sess)