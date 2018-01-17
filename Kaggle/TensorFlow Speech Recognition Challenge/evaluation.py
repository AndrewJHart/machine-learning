import neural_network as nn

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"
# Note that we want to write the outputs of our training predictions so we can
# take a look at where we went wrong on them.
OUTPUT_FILE_NAME = 'training_predictions.csv'
batch_count = 1

use_partial = input("Load just part of the data? ").lower()
if use_partial == "yes" or use_partial == "y":
    use_partial = True
else:
    use_partial = False

# Load up our data, all into one dataset. This is done for training as well as possible.
data = nn.NNData(partial=use_partial, split_type=None)

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)

print()
print()
print("======================")
print("======================")
load_request = input("Load previous model? ").lower()
batch_count = int(data.training.length / int(input("There are {0} training data points. How much data should be used in a batch? ".format(data.training.length))))
if load_request == "yes" or load_request == "y":
    nn.load_model(sess, 'spch')
    # Print out our training accuracy and testing accuracy.
    print("Training Accuracy: {}".format(nn.get_train_accuracy(sess, data, batch=batch_count)))
    # print("Testing Accuracy: {}".format(nn.get_test_accuracy(sess, data)))
    print("======================")
    print("======================")
else:
    count = int(input("How many times should this model be tested? "))
    for i in range(count):
        nn.train_summary(sess, data, merged_summary, writer, batch=batch_count)
    # Print out our best f1_score.
    print("Finished training!")
    print("======================")
    print("======================")
    # Print out our training accuracy and testing accuracy.
    print("Training Accuracy: {}".format(nn.get_train_accuracy(sess, data, batch=batch_count)))
    # Save our model to be used for later.
    print('\rSaving model...', end=" ")
    nn.save_model(sess, 'spch')
    print('\rSaving model...Done')

# Finally, save the results of our actual use case.
nn.save_outputs(sess, data, OUTPUT_FILE_NAME, batch=batch_count)
