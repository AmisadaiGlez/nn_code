# -*- coding: utf-8 -*-
import gzip
import cPickle

import tensorflow as tf
import numpy as np


#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#import matplotlib

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

x_data_train, y_data_train = train_set
y_data_train = one_hot(y_data_train.astype(int),10)

x_data_validation, y_data_validation = valid_set
y_data_validation = one_hot(y_data_validation.astype(int),10)

x_data_test, y_data_test = test_set
y_data_test = one_hot(y_data_test.astype(int),10)

# ---------------- Visualizing some element of the MNIST dataset --------------


# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
#
# plt.imshow(x_data_train[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print y_data_train[57]


# TODO: the neural net!!

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start validation...  "
print "----------------------"

batch_size = 20
error_anterior = 1
error_actual = 0
epoch = 0
f = open ("Error.txt", "w")
while error_anterior-error_actual >= 0.02:
    error_anterior = sess.run(loss, feed_dict={x: x_data_validation, y_: y_data_validation})
    for jj in xrange(len(x_data_train) / batch_size):
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error_actual = sess.run(loss, feed_dict={x: x_data_validation, y_: y_data_validation})
    epoch = epoch + 1
    print "Epoch #:", epoch, "Error Actual: ", error_actual
    errorsrt = str(error_actual).replace(".", ",")
    f.write(errorsrt + "\n")
    print "----------------------------------------------------------------------------------"

print "----------------------"
print "   Start test...  "
print "----------------------"

mal = 0
muestras = 0
result = sess.run(y, feed_dict={x: x_data_test})
for b, r in zip(y_data_test, result):
    muestras += 1
    if tf.argmax(b) != tf.argmax(r):
        mal = mal + 1
        print b, "-->", r , "Mal clasificado"
    else:
        print b, "-->", r
    print "----------------------------------------------------------------------------------"
print "Total de muestras: ", muestras
print "Clasifico mal: ", mal
