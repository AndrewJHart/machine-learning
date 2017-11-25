# import neural_network as nn
import tensorflow as tf
import file_reader as fr
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

# Accuracy methods.
def get_accuracy(sess, data, graph):
    model = graph.get_tensor_by_name("accuracy:0")
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    keep_prob = graph.get_tensor_by_name("kP:0")
    result = sess.run(model, feed_dict={x: data.images, y: data.labels, keep_prob: 1.0})
    return result * 100

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

data = fr.DataPoints(CATEGORIES)
print("======================")
print("======================")
print("Training Accuracy: {}".format(get_accuracy(sess, data, graph)))
print("======================")
print("======================")


# Print out our cross-validation accuracy and testing accuracy.
image = input("Enter an image name: ")
while image.lower() != "quit" and image.lower() != "q":
    print("Prediction: {}".format(get_prediction(sess, image, graph)))
    image = input("Enter an image name: ")
