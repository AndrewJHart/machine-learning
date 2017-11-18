import neural_network as nn

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"
TRAINING_FILE_NAME = 'train.csv'
# Note that we want to write the outputs of our training predictions so we can
# take a look at where we went wrong on them.
TESTING_FILE_NAME = 'train.csv'
OUTPUT_FILE_NAME = 'training_predictions.csv'

# Load up our data, splitting it into 3 pieces: 60% training data, 20% cross validation, 20% testing.
data = nn.NNData(TRAINING_FILE_NAME, TESTING_FILE_NAME, split_training=True)

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)
for i in range(10000):
    nn.train_summary(sess, data, merged_summary, writer)

# Finally, save the results of our actual use case.
nn.save_outputs(sess, data, OUTPUT_FILE_NAME)

# Print out our cross-validation accuracy and testing accuracy.
print()
print()
print("======================")
print("======================")
print("Training Accuracy: {}".format(nn.get_train_accuracy(sess, data)))
print("Cross-validation Accuracy: {}".format(nn.get_cv_accuracy(sess, data)))
print("Testing Accuracy: {}".format(nn.get_test_accuracy(sess, data)))
print("======================")
print("======================")
