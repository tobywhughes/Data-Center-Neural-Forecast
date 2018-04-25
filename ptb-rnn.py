"""
Temporary Practice File
Will be deleted

https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_layers = 3
num_batches = total_series_length // batch_size // truncated_backprop_length

def generate_data():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return (x,y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

#init_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
state_per_layer_list = tf.unstack(init_state, axis = 0)
rnn_tuple_state = tuple(
    [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) 
    for idx in range(num_layers)]
)


W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

#input_series = tf.split(axis = 1, num_or_size_splits=truncated_backprop_length, value=batchX_placeholder)
#labels_series = tf.unstack(batchY_placeholder, axis=1)

#cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
#cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

stacked_rnn = []
for _ in range(num_layers):
    one_cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
    stacked_rnn.append(tf.nn.rnn_cell.DropoutWrapper(one_cell, output_keep_prob=0.5))

cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple=True)

#state_series, current_state = tf.contrib.rnn.static_rnn(cell, input_series, initial_state=rnn_tuple_state)
state_series, current_state = tf.nn.dynamic_rnn(cell, inputs=tf.expand_dims(batchX_placeholder, -1), initial_state= rnn_tuple_state)
state_series = tf.reshape(state_series, [-1, state_size])
logits = tf.matmul(state_series, W2) + b2
labels = tf.reshape(batchY_placeholder, [-1])

#logits_series = [tf.matmul(state, W2) + b2 for state in state_series]
logits_series = tf.unstack(tf.reshape(logits, [batch_size, truncated_backprop_length, num_classes ]), axis=1)
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels = labels)
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2,3,1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x, y = generate_data()
        _current_state = np.zeros((num_layers, 2, batch_size, state_size))
        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_ix = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_ix]
            batchY= y[:,start_idx:end_ix]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()