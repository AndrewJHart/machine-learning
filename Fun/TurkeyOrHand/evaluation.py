import neural_network_training as nn
import tensorflow as tf
import file_reader as fr

CATEGORIES = ['live_turkey', 'cooked_turkey', 'hand_turkey', 'hand_palm']

# Define some constants that we'll want to have for our project.
LOG_DIR = "logs_dir"

# Setup our neural network and train it.
sess, merged_summary, writer = nn.setup(LOG_DIR)
for i in range(1000):
    nn.train_summary(sess, merged_summary, writer)

# Print out our cross-validation accuracy and testing accuracy.
print("======================")
print("======================")
print("Training Accuracy: {}".format(nn.get_train_accuracy(sess)))
all_vars = tf.global_variables()
saver = tf.train.Saver(all_vars)
saver.save(sess, './state.ckpt')
# print("Images count: {}".format(len(data.images))
# print("Labels count: {}".format(len(data.labels))
print("======================")
print("======================")
