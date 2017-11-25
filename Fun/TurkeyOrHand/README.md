# Turkey Or Hand
This project is a pretty straightforward adaption of the MNIST code to handle detecting one of four categories of images: A live turkey, a cooked turkey, a turkey art made from someone's hand, and someone's hand (open palmed).

## Model Information
This model is pretty straightforward in that it is the same as the previous MNIST Tutorial project. It has 2 convolutional layers, each doing a 5x5 pooling on them. Then it creates a fully connected layer, performing dropout on the weights with a selected keep probability.

## Purpose
This is just a fun project that I figured I would do for Thanksgiving 2017. Nothing special, and there is no planned long term support (though feel free to file issues still). The webserver portion includes basically no security, so I am not currently running one anywhere for this. The main goal of this is really to just see what it takes to go from an idea to an implemented product that users could use at a bare minimum. While accuracy is not perfect and security is not state of the art, it servers its purpose :)

## FAQ
### What is included?
This project includes scripts to train a model using the evaluation.py script, run the model using the runner.py script, and even host a NodeJS server as a website with the nodejs project and server_script.py where users can submit their own images. The model that I trained was rather overfit, but still seemed to work decently.

### How does this work?
evaluation.py uses the neural_network.py script to create and train a set of images in a training_data folder (where each subdirectory is one of the four categories for labeling). This model can then be saved to a folder, which can in turn be loaded by the runner.py script. This script will load the model and present the accuracy on the training set (I know, not good practice, this is just a quick project). You can then pass in image file names to load and test with.

### How do I run the web server?
After you have trained the model, saved it, and verified it is working, you can navigate into the nodejs directory and set it up by running npm install. This will pull any dependencies that you may need for the NodeJS project. Then, go ahead and run npm start to start the server. By default the server will be on port 3000 so you'll need to navigate to http://localhost:3000. You can change this by setting a variable PORT in your shell environment to your desired port number.