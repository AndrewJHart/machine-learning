import neural_network as nn

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"
TRAINING_FILE_NAME = 'train.csv'
TESTING_FILE_NAME = 'test.csv'
OUTPUT_FILE_NAME = 'results.csv'

# Load up our data but make sure we train on everything.
data = nn.NNData(TRAINING_FILE_NAME, TESTING_FILE_NAME, split_type=None)

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)
nn.load_model(sess, 'dr')

# Finally, save the results of our actual use case.
nn.save_outputs(sess, data, OUTPUT_FILE_NAME, batch=int(data.output.length / 10))

# Acknowledge that we've saved our results.
print()
print()
print("======================")
print("======================")
print("Output saved to: {}".format(OUTPUT_FILE_NAME))
print("======================")
print("======================")
