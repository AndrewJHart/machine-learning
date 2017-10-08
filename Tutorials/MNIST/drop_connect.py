# This is based on the research done at http://cs.nyu.edu/~wanli/dropc/
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Setup our weights and biases and try to prevent them from
# ever getting to 0 (dying) as best we can.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Setup our convolution (with strides of 1).
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Setup our pooling options (2x2 matrix for pooling)
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

# Create our placeholders for our data again.
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Do our first layer, detecting 32 features in a 5x5 patch.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Do our second layer, detecting 64 features in a 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Identify our features before we process them in the actual network.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Finally do the actual neural network readout, but with
# a DropConnect on the weights instead.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
keep_prob = tf.placeholder(tf.float32)
h_weight_drop = tf.nn.dropout(W_fc2, keep_prob) * keep_prob
prediction = tf.matmul(h_fc1, h_weight_drop) + b_fc2

# Now we train the model and evaluate its accuracy.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

# Create our TensorFlow session and initialize all of our variables.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Start by training the model 20,000 rounds grabbing 100 training
# examples during each round.
for _ in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})

# Finally, calculate our accuracy and print it out.
print("======================")
print("======================")
print("======================")
print("======================")
print("Caculating accuracy...")
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("======================")
print("======================")
print("Accuracy: {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}) * 100))
print("======================")
print("======================")
