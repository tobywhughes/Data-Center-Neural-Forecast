import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv1D, LSTM
from keras.layers import concatenate
from keras.engine.topology import Input
from keras import optimizers
from keras.layers import Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import math, pickle, os
from keras import regularizers


np.random.seed()


def read_subset_series(train_name, test_name):
    train = pickle.load(open(train_name, 'rb'))
    test = pickle.load(open(test_name, 'rb'))
    return (train, test)


def validation_split(train, labels):
    shuffle_index = range(len(train))
    train, labels = shuffle(train, labels)
    train_ret = train[:math.floor(len(train) * .8)]
    train_valid = train[math.floor(len(train) * .8):]
    labels_ret = labels[:math.floor(len(train) * .8)]
    labels_valid = labels[math.floor(len(train) * .8):]
    return train_ret, train_valid, labels_ret, labels_valid

def shuffle_sets(train, labels):
    shuffle_index = range(len(train))
    train, labels = shuffle(train, labels)
    return train, labels

def batches(train, labels, size):
    num = len(train) // size
    train, labels = train[:num * size], labels[:num * size]
    for batch in range(0, len(train), size):
        yield train[batch: batch+size], labels[batch:batch+size]

def reshape_input(series, length = 60, features=3):
    num_samples = len(series)
    for i in range(num_samples):
        series[i] = channel_split(series[i])
    series = np.array(series)
    return series.reshape(num_samples, length, features)

def feed_reshape(series_list, length=60, features=3):
    new_series = []
    for series in series_list:
        new_series.append(reshape_input(series, length, features))
    return tuple(entry for entry in new_series)

def rescale(series_list, scaler):
    new_series = []
    for series in series_list:
        new_series.append(scaler.fit_transform(series))
    return tuple(entry for entry in new_series)

def generate_labels(series, window):
    input_ = []
    labels = []
    for subset in series:
        input_.append(subset[:240])
        output = subset[240:]
        labels.append(np.mean(output[:window]))
    return (input_, labels)

def extract_times(series):
    return [[time[1] for time in subset] for subset in series]

def temp_unsplit(series):
    ret_series = series[0]
    for i in series[1:]:
        ret_series.append(i[-1])
    return [[x] for x in ret_series]

def parse_subsets(packet_list, window_size):
    print("Parsing subsets...")
    current_index = 0
    subset_list = []
    while (current_index + window_size) <= len(packet_list):
        subset_list.append(packet_list[current_index:(current_index + window_size)])
        current_index += 1
    return subset_list

def channel_split(train, window = 60):
    channel1 = train[len(train) - 60:]
    channel2_ = train[len(train) - 120:]
    channel3_ = train
    channel2 = []
    channel3 = []
    for i in range(len(channel1)):
        channel2.append(np.mean([channel2_[(i*2)], channel2_[(i * 2) + 1]]))
        channel3.append(np.mean([channel3_[i * 4], channel3_[(i * 4) + 1], channel3_[(i * 4) + 2], channel3_[(i * 4) + 3]]))
    train = []
    for i in range(len(channel1)):
        train.append([channel1[i], channel2[i], channel3[i]])
    return train

scaler = MinMaxScaler(feature_range=(0, 1))
train, test = read_subset_series('.\\subsets\\20100121subset.p', '.\\subsets\\20100225subset.p')
train2, test2 = read_subset_series('.\\subsets\\20100325subsets.p', '.\\subsets\\20100414subset.p')
train = train + test + train2
test = test2
train = extract_times(train)
train = temp_unsplit(train)
test = extract_times(test)
test = temp_unsplit(test)
len_t = len(train)
fit = train+ test
fit = scaler.fit_transform(fit)
train = fit[:len_t]
test = fit[len_t:]
train = parse_subsets(train, 240+64)
test = parse_subsets(test, 240+64)


window_ = 16
train, train_labels = generate_labels(train, window_)
test_input, test_labels = generate_labels(test, window_)
#train, train_validation, train_labels, labels_validation = validation_split(train, train_labels)
#train, train_validation, test_input = (scaler.fit_transform(entry) for entry in [train, train_validation, test_input])


train, test_input = feed_reshape([train, test_input])
print(train.shape)
#train_labels, test_labels, labels_validation = feed_reshape([train_labels, test_labels, labels_validation])

#model.add(LSTM(10, input_shape=(240, 1)))
input_ = Input(shape=(60,3))
conv = Conv1D(15,10, activation='relu', input_shape = (60,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(.00000002477))(input_)
flat_conv = Flatten()(conv)
lstm = LSTM(20, input_shape=(60, 3))(input_)
lstm_drop = Dropout(.5)(lstm)
dense_conv = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(.00000003619))(flat_conv)
dense_lstm = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(.00000003619))(lstm_drop)
flat = concatenate([dense_conv, dense_lstm], axis=1)
#flat = Flatten()(concat)
dense = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(.00000003619))(flat)
out = Dense(1, activation='relu')(dense)
model = Model(inputs=input_, outputs=out)
#model.add(Conv1D(30,15, activation='relu', input_shape = (60,3), padding='same'))
#model.add(Flatten())
#model.add(Dense(30, activation='relu'))
#model.add(Dense(30, activation='relu'))

#model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train, train_labels, epochs=1000, batch_size=200, validation_split=0.1,verbose=2, shuffle=True)
train_predict = model.predict(train)
test_predict = model.predict(test_input)
train_predict = scaler.inverse_transform(train_predict)
train_labels = scaler.inverse_transform([train_labels])
test_predict = scaler.inverse_transform(test_predict)
test_labels = scaler.inverse_transform([test_labels])
trainScore = mean_squared_error(train_labels[0], train_predict[:,0])
print('Train Score: %.2f MSE' % (trainScore))
testScore = mean_squared_error(test_labels[0], test_predict[:,0])
print('Test Score: %.2f MSE' % (testScore))

plt.plot(range(len(test_labels[0])), test_labels[0])
plt.plot(range(len(test_predict[:,0])), test_predict[:,0])
plt.show()
plt.plot(range(len(train_labels[0])), train_labels[0])
plt.plot(range(len(train_predict[:,0])), train_predict[:,0])
plt.show()