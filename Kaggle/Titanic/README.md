# Titanic
This project is for www.kaggle.com 's Titanic: Machine Learning from Disaster starting competition. It's a pretty straightforward example that focuses on binary classification of who survived on the Titanic based on given data. Training data and testing data are not provided, as they should be downloaded from the competitions site (note that this competition is always active for beginners). More information about this can be found at https://www.kaggle.com/c/titanic/

##### Accuracy: 77.990%

## Usage
The project is pretty straightforward to use. After saving your training data as train.csv and test data as test.csv, you can create and save a model by running the evaluation.py script. This will run the training simulation for however many times you want, saving the model that achieves the highest F1 score on the test data split. You can then run runner.py to run the model that was previously saved against your test.csv data.

## About
This project is broken up into 3 parts: The data importation, the neural network, and the runners. The data importation imports all of our data and formats it into a way we can more easily read it, the neural network takes in that data, splits it into a training set, cross-validation set, and training set, and then trains it on a neural network. The runners either run the training, saving the best model (or evaluate a previously saved model) or run the model on actual data. Each of these parts is explained below.

### Data Importation
The data importation script (data.py) reads our data and extracts out the necessary features for training. The features used are as follows:
* Ticket Class
* Title (as extracted from the name)
* Sex
* Age
* Sibling / spouses aboard the Titanic
* Parents / children aboard the Titanic
* Fare
* Deck (as extracted from the cabin)
* Embarked

The following fields are extracted but not used:
* Family Size
* Fare per person

We then store this data into an array of data points, matched up with arrays of who survived and died. Note that we store both of these so we can more nicely train our data (more on that later). Note that our data for the age and fare are normalized as well, so as not to help prevent overfitting.

### Neural Network
The neural network script (neural_network.py) takes in our data and creates (and trains) a model on it. This code has 3 parts to it: A data splitting portion (via the NNData class), our TensorFlow code for utilizing the model, and helper methods for running each part of the neural network.

#### NNData Class
The NNData class is a helper class for taking our data and splitting it into up to 3 parts: A training set, a test set, and a cross-validation set. This split can be adjusted to your liking, but for this code we utilize a 60-40 split (training and testing sets only). This class also creates an output set, used for creating our final results.

#### TensorFlow Code
For our TensorFlow model, we create 3 layers through we we pass our data: An input layer, 1 hidden layer, and an activation layer. We use a softmax function on the input and activation layer outputs, but not the hidden layer. We have 9 features so 9 inputs for the input layer, and use 10 outputs for it. Thus we use 10 inputs and outputs on our hidden layer, and 10 more inputs on our activation layer. We have 2 outputs on our activation, one indicating the probability the person survived and one for the probability that they lived. This allows us to easily use the TensorFlow softmax function (since it provides a probability distribution summing to 1 accross our outputs). Note that we could do this with one output, but it's a bit more code to do so.

We also track our output which we can then read from outside of TensorFlow, and utilize TensorBoard to create nice graphs of our our neural network looks. You can see this as follows:
![TensorFlow Graph](https://github.com/gemisis/machine-learning/blob/master/Kaggle/Titanic/images/graph.png)

There are several other outputs included in this as well, such as histograms of the weights, accuracy graphs, etc.

#### Helper Methods
Several helper methods are included as well. The first is a setup method which decides what directory our TensorBoard logs should be outputted to. Next there are two training methods: One that just trains on the given data, and the other that trains and outputs a summary for each iteration. Note that both of these allow for batching of the data as it is submitted as well.

We then have several methods for calculating the accuracy, cost, and F1 score on our training data, cross-validation data, and test data. We also have a reset method that will undo all previous training. This is heavily utilized in our evaluation.py script in order to train multiple times (more on this later). Next we have a load_model and save_model method, which are used to load and save the current model/previous models. Finally, we have a method for saving our results on the output data set into a CSV file for submission to Kaggle.

### Runner Scripts
The runner scripts are split into two parts: evaluation.py and runner.py.

#### evalution.py
This script takes in our train.csv file, and splits our data (60/40 split) into a training set and a test set. It can then either load a previous model and tell you the accuracy of it, or create a new model and save it. When it creates the new model, it will run several times (as chosen at runtime), in order to get the best results. This is because our weights have a random intialization in order to improve our accuracy. The model that has the best results on the training set is then saved at the end.

Note that we choose the best model based on the best F1 score, NOT the highest accuracy. This is because the F1 score takes into account both precision and recall, providing a much better measurement of our model. Thus we train multiple models and take the model with the highest F1 score as our best model.

#### runner.py
This script loaders our previously created model and runs it on our train.csv file. It then saves the results of what it calculates into results.csv, which can then be submitted to Kaggle for a score.