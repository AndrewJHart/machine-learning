import tensorflow as tf

def read_image(filename, new_size=28):
    contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(contents, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [new_size, new_size])
    # image = tf.reshape(image, [-1, new_size, new_size, 1])
    return image

class DataPoints:
    images = []
    labels = []
    def __init__(self, categories, category_dir="training_data", split_training=True):
        for category in categories:
            for i in range(1, 40):
                self.images.append(read_image(category_dir + "/" + category + "/" + "training_image(" + str(i) + ").jpg"))
                category_val = [0, 0, 0, 0]
                category_val[categories.index(category)] = 1
                self.labels.append(category_val)
