import neural_network_full as nn
import tensorflow as tf
import file_reader as fr

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)

# Print out our cross-validation accuracy and testing accuracy.
image = input("Enter an image name: ")
while image.lower() != "quit" and image.lower() != "q":
    print("Prediction: {}".format(nn.get_prediction(sess, image)))
    image = input("Enter an image name: ")
