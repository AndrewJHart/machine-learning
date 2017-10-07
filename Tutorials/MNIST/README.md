# MNIST
This project contains several versions of the MNIST tutorials from Google's TensorFlow tutorials, including the beginners version, experts version, and 2 custom versions. The details of each one can be seen below.

Tutorial Link: https://www.tensorflow.org/get_started/

## beginner.py
The biginner code is the simplest, and simply passes everything through a typical regression neural network using softmax for the activation function and a gradient descent for cost minimization. More information on this code can be found at https://www.tensorflow.org/get_started/mnist/beginners

## expert.py
The expert code takes our training data and instead passes it through a convolutional neural network. It contains 2 convolutional layers (each with pooling applied), and applies a dropout layer for reducing overfitting. The activation for each layer uses ReLU function. Instead of a gradient descent for cost minimization, we use an Adam Optimizer, and also lower the learning rate from the beginners sample. More information on this code can be found at https://www.tensorflow.org/get_started/mnist/pros

## drop_connect.py
This example takes the research done at http://cs.nyu.edu/~wanli/dropc/ and applies it to the expert version, resulting in a very small increase in accuracy (~0.02% increase). Other samples of better algorithms for MNIST can be found at https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354

## final.py
This takes everything from drop_connect.py and creates a TensorBoard visualization output into the directory mnist_visualization. This provides a bunch of nice charts through which you can view how the CNN is learning over time, as well as visualize the final results of our training. Several tutorials for this can be found starting at https://www.tensorflow.org/get_started/summaries_and_tensorboard