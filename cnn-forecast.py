import tensorflow as tf
import numpy as np
import pickle
from sklearn.utils import shuffle
import math
import os

tf.logging.set_verbosity(tf.logging.INFO)

#DATA
def read_subset_series(train_name, test_name):
    train = pickle.load(open(train_name, 'rb'))
    test = pickle.load(open(test_name, 'rb'))
    return (train, test)

def generate_labels(series, window):
    input = []
    labels = []
    for subset in series:
        times = [time[1] for time in subset]
        input.append(times[:240])
        output = times[240:]
        labels.append(np.mean(output[:window]))
    return (input, labels)

def validation_split(train, labels):
    shuffle_index = range(len(train))
    train, labels = shuffle(train, labels)
    train_ret = train[:math.floor(len(train) * .8)]
    train_valid = train[math.floor(len(train) * .8):]
    labels_ret = labels[:math.floor(len(train) * .8)]
    labels_valid = labels[math.floor(len(train) * .8):]
    return train_ret, train_valid, labels_ret, labels_valid

def batches(train, labels, size):
    num = len(train) // size
    train, labels = train[:num * size], labels[:num * size]
    for batch in range(0, len(train), size):
        yield train[batch: batch+size], labels[batch:batch+size]

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

train, test = read_subset_series('.\\subsets\\20100121subset.p', '.\\subsets\\20100225subset.p')
train, train_labels = generate_labels(train, 64)
test_input, test_labels = generate_labels(test, 64)

train, train_validation, train_labels, labels_validation = validation_split(train, train_labels)

os.makedirs('.\\save', exist_ok=True)

for i in range(1):
    #Hyperparameters
    batch_size = 50
    seq_len = 60
    learning_rate = 0.0001
    epochs = 100

    n_channels = 3

    graph = tf.Graph()

    #Placeholders

    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
        labels = tf.placeholder(tf.float32, name = 'labels')
        keep_prob = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

        conv = tf.layers.conv1d(inputs = inputs, filters = 30, kernel_size = 15, strides = 1, padding = 'same', activation = tf.nn.relu)
        flat = tf.reshape(conv, [-1, 60 * 30])
        dense = tf.layers.dense(inputs = flat, units = 60, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=dense, units = 1, activation=tf.nn.relu)
        cost = tf.reduce_mean(tf.squared_difference(out, labels))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        count = 0

        for e in range(epochs):
            print("Epoch #" + str(e))
            for train_ , labels_ in batches(train, train_labels, batch_size):
                #train_ = [[[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4]]]
                train_ = [channel_split(batch, 60) for batch in train_]
                feed_ = {inputs: train_, labels: labels_, keep_prob: 0.5, learning_rate_: learning_rate}
                loss, _ = sess.run([cost, optimizer], feed_dict = feed_)
                if count % 10 == 0:
                    print("Loss: {:6f}".format(loss),"on #" + str(count) )
                if count % 20 == 0:
                    val_loss = []
                    for validation_train, validation_labels in batches(train_validation, labels_validation, batch_size):
                        validation_train = [channel_split(batch, 60) for batch in validation_train]
                        validation_feed = {inputs: validation_train, labels: validation_labels, keep_prob: 1.0}
                        validation_loss, _ = sess.run([cost, optimizer], feed_dict = validation_feed)
                        val_loss.append(validation_loss)
                    print("Validation Loss: {:6f}".format(np.mean(val_loss)))
                
                count += 1
        saver.save(sess, 'save/cnn-mr.ckpt')

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('save'))
        test_loss = []
        for train_, labels_ in batches(test_input, test_labels, batch_size):
            train_ = [channel_split(batch, 60) for batch in train_]
            feed = {inputs: train_, labels: labels_, keep_prob: 1}
            loss = sess.run(cost, feed_dict=feed)
            test_loss.append(loss)
        print("Test accuracy: {:.6f}".format(np.mean(test_loss)))