import neural_network as nn

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"
TRAINING_FILE_NAME = 'train.csv'
# Note that we want to write the outputs of our training predictions so we can
# take a look at where we went wrong on them.
TESTING_FILE_NAME = 'train.csv'
OUTPUT_FILE_NAME = 'training_predictions.csv'

# Load up our data, splitting it into 3 pieces: 60% training data, 40% testing.
data = nn.NNData(TRAINING_FILE_NAME, TESTING_FILE_NAME, split_type='test', test_percent=0.4)

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)

print()
print()
print("======================")
print("======================")
load_request = input("Load previous model? ").lower()
if load_request == "yes" or load_request == "y":
    nn.load_model(sess, 'titanic')
    # Print out our training accuracy and testing accuracy.
    print("Training Accuracy: {}".format(nn.get_train_accuracy(sess, data)))
    print("Testing Accuracy: {}".format(nn.get_test_accuracy(sess, data)))
    print("======================")
    print("======================")
else:
    cycle_count = input("How many times should this model be tested? ")
    best_accuracy = 0
    for j in range(int(cycle_count)):
        print("Training cycle {}...".format(j + 1), end="")
        nn.reset()
        for i in range(10000):
            nn.train_summary(sess, data, merged_summary, writer)
        accuracy = nn.get_test_accuracy(sess, data)
        print("done.")
        if accuracy > best_accuracy:
            nn.save_model(sess, 'titanic')
            best_accuracy = accuracy
    # Print out our best accuracy.
    print("Best Accuracy: {}".format(best_accuracy))
    print("======================")
    print("======================")

# Finally, save the results of our actual use case.
nn.save_outputs(sess, data, OUTPUT_FILE_NAME)
