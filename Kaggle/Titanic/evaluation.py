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
    best_f1_score = 0
    for j in range(int(cycle_count)):
        nn.reset()
        count = 10000
        for i in range(count):
            nn.train_summary(sess, data, merged_summary, writer)
            progress = int((i / count) * 10)
            print('\rTraining cycle {0}: [{1}{2}] {3}%'.format(j + 1, '#' * progress, ' ' * (10 - progress), round(100 * (i / count)), 2), end=" ")
        f1_score = nn.get_total_f1_score(sess, data)
        print("\nF1 Score: {}".format(f1_score))
        if f1_score > best_f1_score:
            nn.save_model(sess, 'titanic')
            best_f1_score = f1_score
    # Print out our best f1_score.
    print("Best F1 Score: {}".format(best_f1_score))
    print("======================")
    print("======================")

# Finally, save the results of our actual use case.
nn.save_outputs(sess, data, OUTPUT_FILE_NAME)
