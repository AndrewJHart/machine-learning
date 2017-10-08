from data import read_file
from data import get_batch
import matplotlib.pyplot as plt

# Define some constant variables
DATA_FILE_NAME = "train.csv"

# Read our data
data, size = read_file(DATA_FILE_NAME)
indexes, xs, ys = get_batch(data, 0, size)

# Pull out some of the columns for our graphs.
ages = [float(row[0]) for row in xs]
sex = [row[2] for row in xs]
survived = [row[1] for row in ys]

# Start by just looking at our data graphed out.
plt.figure(1)

# Graph for the ages of those who survived
plt.subplot(211)
plt.title("Age Distribution")
plt.hist(ages, facecolor='green', alpha=0.5)

# Graph for the sex of those who survived.
plt.subplot(212)
plt.title("Survivors")
plt.hist(survived, 10)

# The second figures give us some helpful information about
# how the properties of each person compare to other properties.
# TODO 

# Show the plot at the end.
plt.show()