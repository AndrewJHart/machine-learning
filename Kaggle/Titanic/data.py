# We'll need to import csv to read our CSV files.
import csv
import math

# Define some constants that we'll want to have for our project.
FEATURES = 9
OUTPUTS = 2

def read_file(file_name):
    # Start by reading in our CSV files.
    with open(file_name) as file:
        reader = csv.DictReader(file)
        data = list(reader)
        size = len(data)
        return data, size

gender_option = {'' : 0, 'male' : 1, 'female' : 2}
embarked_options = {'' : 0, 'C' : 1, 'Q' : 2, 'S' : 3}
deck_options = ['Unknown', 'A', 'B', 'C', 'D', 'E', 'F', 'T', 'G']
title_options = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                 'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                 'Don', 'Jonkheer']

def title_ind(name):
    for title in title_options:
        if name.find(title) != -1:
            return title_options.index(title)
    return 'nan'

def deck_ind(cabin):
    for deck in deck_options:
        if cabin.find(deck) != -1:
            return deck_options.index(deck)
    return 0

def get_batch(input_data, start, end):
    output = []
    batch_xs = []
    batch_ys = []
    cleaned_data = get_cleaned(input_data, start, end)

    # We want to do mean normalization on some of our data, so start
    # by calculating the average of our ages and fares.
    avg_age = 0
    avg_fare = 0
    for data_point in cleaned_data:
        avg_age += data_point['age']
        avg_fare += data_point['fare']
    avg_age /= end - start
    avg_fare /= end - start

    # Next calculate our standard deviations for our ages and fares.
    # We'll actually store these when we do the final pass through the data.
    std_age = 0
    std_fare = 0
    for data_point in cleaned_data:
        std_age += (data_point['age'] - avg_age) * (data_point['age'] - avg_age)
        std_fare += (data_point['fare'] - avg_fare) * (data_point['fare'] - avg_fare)
    std_age = math.sqrt((1 / (end - start)) * std_age)
    std_fare = math.sqrt((1 / (end - start)) * std_fare)

    for data_point in cleaned_data:
        # Create a list of outputs that contains the passenger's ID.
        output.append([data_point['passenger_id']])
        # Then create our batch outputs.
        batch_xs.append([(data_point['age'] - avg_age) / std_age, data_point['p_class'], data_point['sex'], 
                         (data_point['fare'] - avg_fare) / std_fare, data_point['sibsp'], data_point['parch'], 
                         data_point['embarked'], data_point['title'], data_point['deck']])
        # Note that we have 2 outputs we expect: One for dead and one for alive.
        # This is done to make sure our activation function works as expected because
        # it ultimately needs to sum to one. See README for more details.
        batch_ys.append([1 - data_point['survived'], data_point['survived']])
    return output, batch_xs, batch_ys

def get_cleaned(input_data, start, end):
    output = []
    for i in range(start, end):
        data = input_data[i]

        # Ultimately we'd like to have our algorithm look at
        # each feature, but we do want to make sure that non of the features
        # would hurt us instead.
        passenger_id = data['PassengerId']              # Used for output
        if 'Survived' in data:
            survived = float(data['Survived'] or 0)     # Used for output
        else:
            survived = 0                                # Used if test data.
        p_class = float(data['Pclass'])                 # Used
        name = data['Name']                             # Used in title
        title = title_ind(name)                         # Used
        sex = gender_option[data['Sex']]                # Used                      * Could be missing in dataset
        age = float(data['Age'] or 0)                   # Used                      * Could be missing in dataset
        sibsp = float(data['SibSp'])                    # Used
        parch = float(data['Parch'])                    # Used
        ticket = data['Ticket']                         # Unused
        fare = float(data['Fare'] or 0)                 # Used                      * Could be missing in dataset
        cabin = data['Cabin']                           # Used in deck
        deck = deck_ind(cabin)                          # Used
        embarked = embarked_options[data['Embarked']]   # Used                      * Could be missing in dataset
        family_size = sibsp + parch + 1                 # Used
        age_class = age * p_class                       # Unused
        fare_per_person = fare / family_size            # Used

        # Create our outputs.
        output.append({'age' : age, 'p_class' : p_class, 'sex' : sex, 'fare' : fare, 
            'sibsp' : sibsp, 'parch' : parch, 'family_size' : family_size, 
            'fare_per_person' : fare_per_person, 'embarked' : embarked, 
            'survived' : survived, 'passenger_id' : passenger_id, 'name' : name,
            'title' : title, 'ticket': ticket, 'cabin': cabin, 'deck': deck})
    return output
