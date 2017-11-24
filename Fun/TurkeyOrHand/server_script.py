import sys
import neural_network as nn
import tensorflow as tf
import file_reader as fr

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)
nn.load_model(sess, 'turkey_day')

# Print out our cross-validation accuracy and testing accuracy.

def main():
    lines = sys.stdin.readlines()
    image = lines[0]
    result, pred = nn.get_prediction(sess, image)
    print(result)

if __name__ == '__main__':
    main()
