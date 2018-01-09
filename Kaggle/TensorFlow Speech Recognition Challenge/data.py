# We'll need to import csv to read our CSV files.
import csv
import math

# Define some constants that we'll want to have for our project.
FEATURES = 784
OUTPUTS = 10

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

        # Add the image ID to our output (starts at 1, so add 1 to the result)
        output.append([i + 1])

        # Create an array of 0s and then set which classification label is our
        # expected result. Then append this to our batched Y outputs.
        ys = [0] * OUTPUTS
        if 'label' in data:
            ys[int(data['label'])] = 1
        batch_ys.append(ys)

        # Finally we go through and actually get our image pixel data.
        # This is currently just quickly hacked together by getting each
        # pixel via its column header.
        pixels = [0] * FEATURES
        for y in range(0, 28):
            for x in range(0, 28):
                pixel = x + y * 28
                pixels[pixel] = data['pixel' + str(pixel)]
        batch_xs.append(pixels)

    return output, batch_xs, batch_ys
