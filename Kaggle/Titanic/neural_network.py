# We'll need to import csv to read our CSV files.
import csv
import tensorflow as tf

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"
TRAINING_FILE_NAME = 'train.csv'
TESTING_FILE_NAME = 'train.csv'
OUTPUT_FILE_NAME = 'results.csv'
BATCH_SIZE = 50
FEATURES = 4
OUTPUTS = 2

# Start by reading in our CSV files.
with open(TRAINING_FILE_NAME) as file:
	reader = csv.DictReader(file)
	training_data = list(reader)
	training_data_size = len(training_data)
with open(TESTING_FILE_NAME) as file:
	reader = csv.DictReader(file)
	testing_data = list(reader)
	testing_data_size = len(testing_data)

# Setup some helper methods.
def weight_variable(shape):
	initial = tf.zeros(shape)
	return tf.Variable(initial, name="W")

def bias_variable(shape):
	initial = tf.zeros(shape)
	return tf.Variable(initial, name="b")

def get_batch(input_data, start, end):
	indexes = []
	batch_xs = []
	batch_ys = []
	for i in range(start, end):
		data = input_data[i]

		passenger_id = data['PassengerId']
		survived = float(data['Survived'])
		p_class = float(data['Pclass'])
		name = data['Name']
		sex = 0 if data['Sex'] == 'male' else 1
		age = data['Age'] or 0
		sibsp = data['SibSp']
		parch = data['Parch']
		ticket = data['Ticket']
		fare = data['Fare'] or 0
		cabin = data['Cabin']
		embarked = data['Embarked']

		indexes.append([passenger_id, name])

		batch_xs.append([age, p_class, sex, fare])
		batch_ys.append([1 - survived, survived])
	return indexes, batch_xs, batch_ys

# Next lets go ahead and calculate what our results are.
with tf.name_scope("prediction"):
	x = tf.placeholder(tf.float32, [None, FEATURES], name="inputs")
	y_ = tf.placeholder(tf.float32, [None, OUTPUTS], name="actuals")
	W = weight_variable([FEATURES, OUTPUTS])
	b = bias_variable([OUTPUTS])
	y_mid = tf.matmul(x, W) + b
	y = tf.nn.softmax(y_mid)
	output = tf.argmax(y, 1)
	# Give some summaries for the outputs.
	tf.summary.histogram("weights", W)
	tf.summary.histogram("biases", b)
	tf.summary.histogram("activation", y)

# Now calculate the error and train it.
with tf.name_scope("cost"):
	cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y + 1e-10), reduction_indices=1))
	tf.summary.scalar("cost", cost)
with tf.name_scope("train"):
	train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# Calculate the accuracy finally.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

# Setup our session.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_DIR)
writer.add_graph(sess.graph)

# Train everything.
for j in range(10000):
	for i in range(0, int(training_data_size / BATCH_SIZE)):
		indexes, training_xs, training_ys = get_batch(training_data, i * BATCH_SIZE, i * BATCH_SIZE + BATCH_SIZE)
		# print(sess.run(y, feed_dict={x: training_xs, y_: training_ys}))
		s, t = sess.run([merged_summary, train_step], feed_dict={x: training_xs, y_: training_ys})
		if j % 500 == 0:
			writer.add_summary(s, j + i * (10000 / 500))

# Finally, test our accuracy and print out stats about how well this model did.
indexes, test_xs, test_ys = get_batch(testing_data, 0, testing_data_size)
print()
print()
print("======================")
print("======================")
print("Data size: {} Batch size: {}".format(training_data_size, BATCH_SIZE))
print("Accuracy: {}".format(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}) * 100))
print("======================")
print("======================")

with open("results.csv", "w") as out_file:
	out_file.write("PassengerId,Survived,Name\n")
	output = sess.run(output, feed_dict={x: test_xs, y_: test_ys})
	for index, prediction in zip(indexes, output):
		out_file.write("{0},{1},{2}\n".format(index[0], prediction, index[1]))
