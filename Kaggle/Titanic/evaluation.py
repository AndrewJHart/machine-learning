import neural_network as nn

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"
TRAINING_FILE_NAME = 'train.csv'
TESTING_FILE_NAME = 'test.csv'
OUTPUT_FILE_NAME = 'results.csv'

# Load up our data, splitting it into 3 pieces: 60% training data, 20% cross validation, 20% testing.
data = nn.NNData(TRAINING_FILE_NAME, TESTING_FILE_NAME, split_training=True)

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)
for i in range(1000):
    nn.train_summary(sess, data, merged_summary, writer, batch=10)

# Print out our cross-validation accuracy and testing accuracy.
print()
print()
print("======================")
print("======================")
print("Cross-validation Accuracy: {}".format(nn.get_cv_accuracy(sess, data)))
print("Testing Accuracy: {}".format(nn.get_test_accuracy(sess, data)))
print("======================")
print("======================")
