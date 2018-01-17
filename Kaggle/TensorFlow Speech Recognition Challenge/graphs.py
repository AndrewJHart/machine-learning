from data import read_file
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define some constant variables
DATA_FILE_NAME = "data/test/audio/clip_006b6248b.wav"

# Read our data
data, size, sample_rate = read_file(DATA_FILE_NAME)

print("Size: {0}; Sample Rate: {1}".format(size, sample_rate))

# Start with a 3d graph show the survivors comparing fare, family size, and age.
figure_a = plt.figure()
frequency_graph = figure_a.add_subplot(111)
frequency_graph.set_title("Frequency", y=1.08)
frequency_graph.set_xlabel('Sample Number')
frequency_graph.set_ylabel('Amplitude')
frequency_graph.plot(data)

# # Let's do a second 3D graph with Siblings/parents, Parents/children, and age.
# figure_b = plt.figure()
# sibsp_parch_age = figure_b.add_subplot(111, projection='3d')
# sibsp_parch_age.set_title("SibsP vs ParCh vs Age", y=1.08)
# sibsp_parch_age.set_xlabel('SibsP')
# sibsp_parch_age.set_ylabel('ParCh')
# sibsp_parch_age.set_zlabel('Age')
# for survivor in survivors:
#     sibsp_parch_age.scatter(survivor['sibsp'], survivor['parch'], survivor['age'], c='g', marker='o')
# for dead in deaths:
#     sibsp_parch_age.scatter(dead['sibsp'], dead['parch'], dead['age'], c='r', marker='^')

# # Do a 2D plot of Fare vs Age this time around.
# figure_c = plt.figure()
# age_fare_surv = figure_c.add_subplot(111)
# age_fare_surv.set_title("Fare vs Age", y=1.08)
# age_fare_surv.set_xlabel('Fare')
# age_fare_surv.set_ylabel('Age')
# for survivor in survivors:
#     age_fare_surv.plot(survivor['fare'], survivor['age'], c='g', marker='o')
# for dead in deaths:
#     age_fare_surv.plot(dead['fare'], dead['age'], c='r', marker='^')

# # Next do a 2D plot of Fare vs Family Size this time around.
# figure_d = plt.figure()
# family_size_fare_surv = figure_d.add_subplot(111)
# family_size_fare_surv.set_title("Fare vs Family Size", y=1.08)
# family_size_fare_surv.set_xlabel('Fare')
# family_size_fare_surv.set_ylabel('Family Size')
# for survivor in survivors:
#     family_size_fare_surv.plot(survivor['fare'], survivor['family_size'], c='g', marker='o')
# for dead in deaths:
#     family_size_fare_surv.plot(dead['fare'], dead['family_size'], c='r', marker='^')

# # Quick bar graph of embarked survivors from each port.
# figure_e = plt.figure()
# embarked_surv = figure_e.add_subplot(111)
# embarked_surv.set_title("Embarked Counts", y=1.08)
# embarked_surv.set_xlabel('Embarked')
# embarked_surv.set_ylabel('Counts')
# embarked_survived_counts = [0, 0, 0, 0]
# for survivor in survivors:
#     embarked_survived_counts[survivor['embarked']] += 1
# embarked_death_counts = [0, 0, 0, 0]
# for death in deaths:
#     embarked_death_counts[death['embarked']] += 1
# embarked_surv.bar(np.arange(len(embarked_survived_counts)), embarked_survived_counts, 0.25, color='g')
# embarked_surv.bar(np.arange(len(embarked_death_counts)) + 0.25, embarked_death_counts, 0.25, color='r')

# Show the plot at the end.
plt.show()