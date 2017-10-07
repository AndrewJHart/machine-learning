import tensorflow as tf

# Grab our data tp start/
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Input data as placeholders since they'll be filled in when we
# run the model (x being our input images and y being our expected results
# to correct the errors of.) Note that None indicates that dimension can be
# any size.
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Our weights and variables however are variables since
# they will be modified during our calculations when we are training.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Predict our outputs.
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# Calculate our cost for the model (how wrong we were).
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
# Then create a training step that we run to correct this (e.g. minimize the cost amount).
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# Create our TensorFlow session and initialize all of our variables.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Start by training the model 1000 rounds grabbing 100 training
# examples during each round.
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

# Finally, calculate our accuracy and print it out.
# We do this by using tf.argmax to compare values, which
# finds the highest value in our vector and returns its index.
# E.g. If our vector is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] then it returns 3.
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("======================")
print("======================")
print("Accuracy: {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}) * 100))
print("======================")
print("======================")
