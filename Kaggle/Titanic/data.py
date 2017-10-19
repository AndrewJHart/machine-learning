# We'll need to import csv to read our CSV files.
import csv

def read_file(file_name):
    # Start by reading in our CSV files.
    with open(file_name) as file:
        reader = csv.DictReader(file)
        data = list(reader)
        size = len(data)
        return data, size

def get_batch(input_data, start, end):
    output = []
    batch_xs = []
    batch_ys = []
    for i in range(start, end):
        data = input_data[i]

        # Ultimately we'd like to have our algorithm look at
        # each feature, but we do want to make sure that non of the features
        # would hurt us instead.
        passenger_id = data['PassengerId']          # Used for output
        if 'Survived' in data:
            survived = float(data['Survived'] or 0) # Used for output
        else:
            survived = 0                            # Used if test data.
        p_class = float(data['Pclass'])             # Used
        name = data['Name']                         # Unused
        sex = 1 if data['Sex'] == 'male' else 0     # Used
        age = data['Age'] or 0                      # Used
        sibsp = data['SibSp']                       # Used
        parch = data['Parch']                       # Used
        ticket = data['Ticket']                     # Unused
        fare = data['Fare'] or 0                    # Used
        cabin = data['Cabin']                       # Unused
        embarked = data['Embarked']                 # Unused

        # Create a list of outputs that contains the passenger's ID.
        output.append([passenger_id])
        # Then create our batch outputs.
        batch_xs.append([age, p_class, sex, fare, sibsp, parch])
        # Note that we have 2 outputs we expect: One for dead and one for alive.
        # This is done to make sure our softmax activation works as expected because
        # it ultimately needs to sum to one. See README for more details.
        batch_ys.append([1 - survived, survived])
    return output, batch_xs, batch_ys
