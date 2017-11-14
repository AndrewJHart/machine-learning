# We'll need to import csv to read our CSV files.
import csv

# Define some constants that we'll want to have for our project.
FEATURES = 6
OUTPUTS = 2

def read_file(file_name):
    # Start by reading in our CSV files.
    with open(file_name) as file:
        reader = csv.DictReader(file)
        data = list(reader)
        size = len(data)
        return data, size

gender_option = {'' : 0, 'male' : 1, 'female' : 0}
embarked_options = {'' : 0, 'C' : 1, 'Q' : 2, 'S' : 3}
title_options = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                 'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                 'Don', 'Jonkheer']

def title_ind(name):
    for title in title_options:
        if name.find(title) != -1:
            return title_options.index(title)
    return 'nan'

def get_batch(input_data, start, end):
    output = []
    batch_xs = []
    batch_ys = []
    cleaned_data = get_cleaned(input_data, start, end)
    for data_point in cleaned_data:
        # Create a list of outputs that contains the passenger's ID.
        output.append([data_point['passenger_id']])
        # Then create our batch outputs.
        # batch_xs.append([data_point['age'], data_point['p_class'], data_point['sex'], 
        #                  data_point['fare'], data_point['sibsp'], data_point['parch'], data_point['fare_per_person'], 
        #                  data_point['embarked'], data_point['title']])
        batch_xs.append([data_point['age'], data_point['p_class'], data_point['sex'],
            data_point['title'], data_point['sibsp'], data_point['parch']])
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
        cabin = data['Cabin']                           # Unused
        embarked = embarked_options[data['Embarked']]   # Used                      * Could be missing in dataset
        family_size = sibsp + parch + 1                 # Used
        age_class = age * p_class                       # Unused
        fare_per_person = fare / family_size            # Used

        # Create our outputs.
        output.append({'age' : age, 'p_class' : p_class, 'sex' : sex, 'fare' : fare, 
            'sibsp' : sibsp, 'parch' : parch, 'family_size' : family_size, 
            'fare_per_person' : fare_per_person, 'embarked' : embarked, 
            'survived' : survived, 'passenger_id' : passenger_id, 'name' : name,
            'title' : title})
    return output
