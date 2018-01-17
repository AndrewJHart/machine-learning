import neural_network as nn

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"
# Note that we want to write the outputs of our training predictions so we can
# take a look at where we went wrong on them.
OUTPUT_FILE_NAME = 'results.csv'
batch_count = 1

use_partial = input("Load just part of the data? ").lower()
if use_partial == "yes" or use_partial == "y":
    use_partial = True
else:
    use_partial = False

# Load up our data, all into one dataset. This is done for training as well as possible.
data = nn.NNData(partial=use_partial, split_type=None, load_training_data=False)

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)

print()
print()
print("======================")
print("======================")
batch_count = int(data.training.length / int(input("There are {0} data points. How much data should be used in a batch? ".format(data.training.length))))
nn.load_model(sess, 'spch')
print("======================")
print("======================")

# Finally, save the results of our actual use case.
nn.save_outputs(sess, data, OUTPUT_FILE_NAME, batch=batch_count, include_actual=False)

# Acknowledge that we've saved our results.
print()
print()
print("======================")
print("======================")
print("Output saved to: {}".format(OUTPUT_FILE_NAME))
print("======================")
print("======================")
