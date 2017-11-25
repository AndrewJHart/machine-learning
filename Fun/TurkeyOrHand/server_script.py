# import neural_network as nn
import tensorflow as tf
import file_reader as fr
import sys
import os

# Define some constants that we'll want to have for our project.
CATEGORIES = ['live_turkey', 'cooked_turkey', 'hand_turkey', 'hand_palm']

# Helper method for reloading the model.
def load_model(sess, directory="model"):
    if os.path.exists(directory):
        tf.saved_model.loader.load(sess, ["tag"], directory)
        return tf.get_default_graph()
    else:
        print("Error! Model does not exist!")
        exit(-1)

def get_prediction(sess, image, graph):
    image_data = fr.read_image(image)
    model = graph.get_tensor_by_name("predicted_category:0")
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    keep_prob = graph.get_tensor_by_name("kP:0")
    out = sess.run(model, feed_dict={x: [image_data], y: [[0, 0, 0, 0]], keep_prob: 1.0})
    name = CATEGORIES[out[0]]
    return name

# Setup our session and load in our model.
sess = tf.InteractiveSession()
graph = load_model(sess)

# Print out our cross-validation accuracy and testing accuracy.
def main():
    lines = sys.stdin.readlines()
    image = lines[0]
    result = get_prediction(sess, image, graph)
    print(result)

if __name__ == '__main__':
    main()
