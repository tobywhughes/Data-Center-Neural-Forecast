import numpy as np 
import matplotlib.pyplot as plt
import pickle

def read_subset_series(file_name):
    train = pickle.load(open(file_name, 'rb'))
    return train

def generate_labels(series, window):
    inputs = []
    labels = []
    for subset in series:
        times = [time[1] for time in subset]
        inputs.append(times[:240])
        output = times[240:]
        labels.append(np.mean(output[:window]))
    return (inputs, labels)


def generate_predictions(subsets):
    predictions = []
    for subset in subsets:
        predictions.append(subset[-1])
    return predictions

def squared_error(predictions, labels):
    errors = []
    for i in range(len(predictions)):
        error = labels[i] - predictions[i]
        errors.append(error ** 2)
    return errors

def mean_squared_error_by_window(series, window):
    inputs, labels = generate_labels(series, window)
    predictions = generate_predictions(inputs)
    sq_error = squared_error(predictions, labels)
    return np.mean(sq_error)

series = read_subset_series('.\\subsets\\20100414subset.p')
windows = [2 ** window for window in range(7)]
mse_values = [mean_squared_error_by_window(series, window) for window in windows]
print(mse_values)
plt.plot(windows, mse_values)
plt.axis([0, windows[-1], 0, max(mse_values) * 1.1])
plt.show()