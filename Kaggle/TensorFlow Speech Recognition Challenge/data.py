# We'll need to import csv to read our CSV files.
import librosa
import numpy as np
import math
import os

# Define some constants that we'll want to have for our project.
TRAINING_PATH_BASE = "data/train/audio/"
TEST_PATH = "data/test/audio/"
FEATURES = 22050
OUTPUTS = 12
RESULT_MAP = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']

commands = {'yes' : [], 'no' : [], 'up' : [], 'down' : [], 'left' : [], 'right' : [], 'on' : [], 'off' : [], 'stop' : [], 'go' : []}
unknowns = {'bed' : [], 'bird' : [], 'cat' : [], 'dog' : [], 'eight' : [], 'five' : [], 'four' : [], 'happy' : [], 'house' : [], 'marvin' : [], 'nine' : [], 'one' : [], 'seven' : [], 'sheila' : [], 'six' : [], 'three' : [], 'tree' : [], 'two' : [], 'wow' : [], 'zero' : []}
background_noise = {'_background_noise_' : []}
training_file_count = 0
test_files = []

print("Loading training files...")
for key, files in commands.items():
    path = TRAINING_PATH_BASE + key + "/"
    for file in os.listdir(path):
        if (file.endswith('.wav')):
            files.append([file, os.path.join(path, file)])
            training_file_count += 1

for key, files in unknowns.items():
    path = TRAINING_PATH_BASE + key + "/"
    for file in os.listdir(path):
        if (file.endswith('.wav')):
            files.append([file, os.path.join(path, file)])
            training_file_count += 1

for key, files in background_noise.items():
    path = TRAINING_PATH_BASE + key + "/"
    for file in os.listdir(path):
        if (file.endswith('.wav')):
            files.append([file, os.path.join(path, file)])
            training_file_count += 1

print("{0} training files found!".format(training_file_count))

print("Loading test files...")
for file in os.listdir(TEST_PATH):
    test_files.append([file, os.path.join(TEST_PATH, file)])
print("{0} test files found!".format(len(test_files)))

# Reads our audio file. Note that we expect it to be 1 second long.
# Some files are not however so we pad them to ensure they are uniform.
def read_file_no_extras(file_name):
    # Load our audio file and record some info about it.
    waveform, sample_rate = librosa.load(file_name)
    length = len(waveform)
    # Pad the end of it if it is shorter than expected.
    if length < sample_rate:
        waveform = np.pad(waveform, (0, sample_rate - length), 'constant')
        length = len(waveform)
    # Remove the end if it is longer than expected.
    if length > sample_rate:
        waveform = waveform[0:sample_rate]
        length = len(waveform)
    return waveform, length, sample_rate

# Reads our audio file. Note that we expect it to be 1 second long.
# Some files are not however so we pad them to ensure they are uniform.
def read_file(file_name):
    # Load our audio file and record some info about it.
    waveform, sample_rate = librosa.load(file_name)
    length = len(waveform)
    # Pad the end of it if it is shorter than expected.
    if length < sample_rate:
        waveform = np.pad(waveform, (0, sample_rate - length), 'constant')
        length = len(waveform)
        waveform = [waveform]
        length = [length]
    # Remove the end if it is longer than expected.
    elif length > sample_rate:
        results = []
        for i in range(0, length, sample_rate):
            if (i + sample_rate <= length):
                results.append(waveform[i:i + sample_rate])
            else:
                temp = waveform[i:length]
                temp = np.pad(temp, (0, sample_rate - (length - i)), 'constant')
                results.append(temp)
        waveform = results
        length = [sample_rate]
    else:
        waveform = [waveform]
        length = [length]
    return waveform, length, sample_rate

def get_files(partial_load=False, training_data=True):
    xs = []
    ys = []
    filenames = []
    progress = 0
    files_loaded = 0
    print('\rLoading files: [{0}{1}] {2}%'.format('#' * progress, ' ' * (50 - progress), round(50 * progress, 2)), end=" ")

    if training_data:
        for key, files in commands.items():
            for file in files:
                waveforms, length, sample_rate = read_file(file[1])
                for waveform in waveforms:
                    xs.append(waveform)
                    result = [0] * len(RESULT_MAP)
                    result[RESULT_MAP.index(key)] = 1
                    ys.append(result)
                    filenames.append(file[0])

                # Update progress bar.
                files_loaded += 1
                progress = files_loaded / training_file_count * 50
                print('\rLoading files: [{0}{1}] {2}%'.format('#' * int(progress), ' ' * (50 - int(progress)), round(progress / 50 * 100, 2)), end=" ")
                # Allow partial loading for speed.
                if partial_load and progress / 50 * 100 > 0.1:
                    break

        for key, files in background_noise.items():
            for file in files:
                waveforms, length, sample_rate = read_file(file[1])
                for waveform in waveforms:
                    xs.append(waveform)
                    result = [0] * len(RESULT_MAP)
                    result[RESULT_MAP.index('silence')] = 1
                    ys.append(result)
                    filenames.append(file[0])

                # Update progress bar.
                files_loaded += 1
                progress = files_loaded / training_file_count * 50
                print('\rLoading files: [{0}{1}] {2}%'.format('#' * int(progress), ' ' * (50 - int(progress)), round(progress / 50 * 100, 2)), end=" ")
                # Allow partial loading for speed.
                if partial_load and progress / 50 * 100 > 0.2:
                    break

        for key, files in unknowns.items():
            for file in files:
                waveforms, length, sample_rate = read_file(file[1])
                for waveform in waveforms:
                    xs.append(waveform)
                    result = [0] * len(RESULT_MAP)
                    result[RESULT_MAP.index('unknown')] = 1
                    ys.append(result)
                    filenames.append(file[0])

                # Update progress bar.
                files_loaded += 1
                progress = files_loaded / training_file_count * 50
                print('\rLoading files: [{0}{1}] {2}%'.format('#' * int(progress), ' ' * (50 - int(progress)), round(progress / 50 * 100, 2)), end=" ")
                # Allow partial loading for speed.
                if partial_load and progress / 50 * 100 > 0.2:
                    break
    else:
        for file in test_files:
            waveform, length, sample_rate = read_file_no_extras(file[1])
            xs.append(waveform)
            result = [0] * len(RESULT_MAP)
            # result[RESULT_MAP.index(key)] = 1
            ys.append(result)
            filenames.append(file[0])

            # Update progress bar.
            files_loaded += 1
            progress = files_loaded / len(test_files) * 50
            print('\rLoading files: [{0}{1}] {2}%'.format('#' * int(progress), ' ' * (50 - int(progress)), round(progress / 50 * 100, 2)), end=" ")
            # Allow partial loading for speed.
            if partial_load and progress / 50 * 100 > 0.1:
                break

    return xs, ys, filenames