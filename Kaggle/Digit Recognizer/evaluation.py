import neural_network as nn

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"
TRAINING_FILE_NAME = 'train.csv'
# Note that we want to write the outputs of our training predictions so we can
# take a look at where we went wrong on them.
TESTING_FILE_NAME = 'train.csv'
OUTPUT_FILE_NAME = 'training_predictions.csv'
batch_count = 1

# Load up our data, all into one dataset. This is done for training as well as possible.
data = nn.NNData(TRAINING_FILE_NAME, TESTING_FILE_NAME, split_type=None)

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)

print()
print()
print("======================")
print("======================")
load_request = input("Load previous model? ").lower()
if load_request == "yes" or load_request == "y":
    nn.load_model(sess, 'dr')
    # Print out our training accuracy and testing accuracy.
    print("Training Accuracy: {}".format(nn.get_train_accuracy(sess, data)))
    print("Testing Accuracy: {}".format(nn.get_test_accuracy(sess, data)))
    print("======================")
    print("======================")
else:
    count = int(input("How many times should this model be tested? "))
    batch_count = int(data.training.length / int(input("There are {0} training data points. How much data should be used in a batch? ".format(data.training.length))))
    for i in range(count):
        nn.train_summary(sess, data, merged_summary, writer, batch=batch_count)
    # Print out our best f1_score.
    print("Finished training!")
    print("======================")
    print("======================")

# Finally, save the results of our actual use case.
nn.save_outputs(sess, data, OUTPUT_FILE_NAME, batch_count)
