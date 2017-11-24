import neural_network as nn
import tensorflow as tf
import file_reader as fr

CATEGORIES = ['live_turkey', 'cooked_turkey', 'hand_turkey', 'hand_palm']

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"
data = fr.DataPoints(CATEGORIES)

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)
for i in range(1000):
    nn.train_summary(sess, data, merged_summary, writer)

# Print out our cross-validation accuracy and testing accuracy.
print("======================")
print("======================")
print("Training Accuracy: {}".format(nn.get_accuracy(sess, data)))
save_request = input("Save this model? ")
if save_request.lower() == "yes" or save_request.lower() == "y":
    print("saving...")
    nn.save_model(sess, 'turkey_day')
    print("successfully saved!")
# print("Images count: {}".format(len(data.images))
# print("Labels count: {}".format(len(data.labels))
print("======================")
print("======================")
