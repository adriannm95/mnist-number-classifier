import gzip
import cPickle

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
        x = np.array(x) #matriz de numpy
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()



train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)



# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

# Esto hace referencia a la entrada
W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

# Esto hace referencia a las salidas de las neuronas
W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)


#formula del sigmoide (suma de las entradas y la salida debe ser 0 o 1

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)#normalizamos los valores de la matriz [0,0,0,0,1,0,0,0]

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)



print "----------------------"
print "   Start training...  "
print "----------------------"



batch_size = 20

train_graphic = []
valid_graphic = []

epoch=0

for epoch in xrange(10):
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})


    #Errores del entrenamiento

    error_graphic = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    train_graphic.append(error_graphic)


    print "Validacion"

    print "Epoca: ",epoch , "Error", error_graphic

    valid_data= sess.run(loss, feed_dict={x: valid_x, y_: valid_y})




print "Test realizados"
nerrores=0
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    if np.argmax(b) != np.argmax(r):
        #posicion en b es distinta de r (mayores elementos ambos) se debe tratar de un error
        # print "-----> ERROR"
        nerrores += 1
#print 'Porcentaje de errores %.4f%%' % (nerrores/len(result))




plt.plot(train_graphic)
plt.show()

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!
