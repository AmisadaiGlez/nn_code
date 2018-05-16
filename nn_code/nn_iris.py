import tensorflow as tf
import numpy as np


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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

x_data_train = x_data[:105]
y_data_train = y_data[:105]

x_data_validation = x_data[106:128]
y_data_validation = y_data[106:128]

x_data_test = x_data[129:]
y_data_test = y_data[129:]

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

"""

*** Estudio del conjunto de datos sin dividir:

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
mal = 0
for epoch in xrange(100):
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        if np.argmax(b) != np.argmax(r):
            mal = mal + 1
            print b, "-->", r, "Mal clasificado"
        else:
            print b, "-->", r
        print "----------------------------------------------------------------------------------"

print "Clasifico mal: ",mal

*** El resultado es que se obtienen mas de 250 errores en las ejecuciones a la hora de clasificar.

"""

print "----------------------"
print "   Start validation...  "
print "----------------------"

batch_size = 20

f = open ("Error.txt", "w")

for epoch in xrange(323):
    for jj in xrange(len(x_data_train) / batch_size):
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: x_data_validation, y_: y_data_validation})
    print "Epoch #:", epoch, "Error: ", error
    errorsrt = str(error).replace(".", ",")
    f.write(errorsrt + "\n")

    print "----------------------------------------------------------------------------------"

print "----------------------"
print "   Start test...  "
print "----------------------"

mal = 0
result = sess.run(y, feed_dict={x: x_data_test})
for b, r in zip(y_data_test, result):
    if np.argmax(b) != np.argmax(r):
        mal = mal + 1
        print b, "-->", r , "Mal clasificado"
    else:
        print b, "-->", r
    print "----------------------------------------------------------------------------------"

print "Clasifico mal: ", mal

"""

A la hora de clasificar sobre el conjunto de test se obtiene un resultado muy inferior al obtenido intentando clasificar
todo el conjunto de datos, con todo el conjunto se producian mas de 250 malas clasificaciones, y ahora con el conjunto 
de test se consigue reducir ese numero hasta 2 o menos malas clasificaciones 

"""