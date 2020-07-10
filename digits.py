from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os
# import pil and skimage
from PIL import Image
# for some reason, the following is needed to run on mac os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn import datasets
digits = datasets.load_digits()

# import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences



# from keras import backend
# from tensorflow.keras import backend





x = digits.images.reshape((len(digits.images), -1))
x.shape


from keras.utils import np_utils
y = np_utils.to_categorical(digits.target,10)
print(digits.target)
print(y)

split_limit=1000
x_train = x[:split_limit]
y_train = y[:split_limit]
x_test = x[split_limit:]
y_test = y[split_limit:]

from keras import layers, Model, optimizers, regularizers
# create the input layer
#
# we specify that the input layer
# should have 64 neurons, one for each pixel
# in our images.
# The input neurons do nothing, they
# just transfer the value at each pixel
# to the next layer.
img_input = layers.Input(shape=(64,))

# create the hidden layer
#
# This layer is a Dense layer, which means
# that its neurons are fully connected to the
# neurons in the previous layer (the input layer)
# We will talk about the activation in a future post
tmp = layers.Dense(15,
                   activation='sigmoid')(img_input)

# create the output layer
#
# The output layer is another Dense layer.
# It must have 10 neurons, corresponding to
# the 10 digit categories
# output = layers.Dense(10,
#                       activation='softmax')(tmp)

output = layers.Dense(10,
                      activation='softmax')(img_input)

# create the neural network from the layers
model = Model(img_input, output)

# print a summary of the model
model.summary()

# =================================================
# Please don't pay attention to what follows,
# we'll talk about regularization later!
# For now, it is enough to know that regularization
# helps the neural network converge properly.
# I've added this regularization because it is
# performed by default in scikit-learn,
# and because we want to be able to compare the
# results of scikit-learn and keras.
l2_rate = 1e-4
for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = regularizers.l2(l2_rate)
        layer.bias_regularizer = regularizers.l2(l2_rate)
        layer.activity_regularizer = regularizers.l2(l2_rate)
# =================================================

# define how the neural network will learn,
# and compile the model.
# models must be compiled before
# they can be trained and used.
# the loss, optimizer, and metrics arguments
# will be covered in a future post.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.1, momentum=0.9),
              metrics=['accuracy'])

history = model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test),
                    batch_size=100, epochs=50)

predictions = model.predict(x_test)
print(predictions[3])


def plot_prediction(index):
    print('predicted probabilities:')
    print(predictions[index])
    print('predicted category', np.argmax(predictions[index]))
    print('true probabilities:')
    print(y_test[index])
    print('true category', np.argmax(y_test[index]))
    img = x_test[index].reshape(8,8)
    plt.imshow(img)


plot_prediction(3)


# the second argument of argmax specifies
# that we want to get argmax for each example.
# without this argument, argmax would return
# the largest value in the whole array,
# considering all examples
y_test_best = np.argmax(y_test,1)
print(y_test_best.shape)
predictions_best = np.argmax(predictions,1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test_best, predictions_best)


def see(image):
    """See an image based on a np array.

    Arguments:
        image: numpy array

    Returns:
        None
    """

    # check if it is a shadow
    if len(image.shape) < 3:
        # reproject
        image = holograph(image)

    # view the image
    Image.fromarray(image).show()

    return None


def holograph(shadow):
    """Convert a two dimension shadow to a three dimensional grayscale image.

    Arguments:
        shadow: np.array

    Returns:
        np.array
    """

    # expanding into rgb function
    expanding = lambda gray: [int(gray * 255), int(gray * 255), int(gray * 255), 255]

    # construct hologram
    hologram = [[expanding(entry + 0.5) for entry in row] for row in shadow]
    hologram = np.array(hologram,  dtype=np.uint8)

    return hologram


# make session
# graph = tensorflow.Graph()
# s#ession = tensorflow.Session(graph=graph)
# session.run(tensorflow.global_variables_initializer())
# weights = self.model.layers[1].weights[0]
def probe(index):

    evaluation = model.layers[1].get_weights()[0]
    print(evaluation.shape)
    neurons = [neuron for neuron in zip(*evaluation)]
    neuron = neurons[index]

    # get max and min
    minimum = min(neuron)
    maximum = max(neuron)
    spread = maximum - minimum

    # normalize
    normalization = [((entry - minimum) / spread) - 0.5 for entry in neuron]

    # break into rows
    length = int(len(normalization) / 8)

    # make image
    image = np.array([normalization[index * length: (index + 1) * length] for index in range(8)])
    see(image)

    return None





#
# evaluationii = model.layers[2].get_weights()[0]
# biases = model.layers[2].get_weights()[1]
#
# print(evaluation.shape)
# print(evaluationii.shape)
# # session.close()
#
# # separate per neuron
# evaluation = evaluation.tolist()
# neurons = [neuron for neuron in zip(*evaluation)]
# neuronsii = [neuron for neuron in zip(*evaluationii)]
#
# # for each target
# for target, bias in zip(neuronsii, biases):
#
#     # for each neuron
#     perception = np.zeros((64,))
#     for index, neuron in enumerate(neurons):
#         perception = np.array(perception) + np.array(neuron) * (target[index] + bias)
#
#     # get max and min
#     minimum = min(perception)
#     maximum = max(perception)
#     spread = maximum - minimum
#
#     # normalize
#     normalization = [((entry - minimum) / spread) - 0.5 for entry in perception]
#
#     # break into rows
#     length = int(len(normalization) / 8)
#
#     # make image
#     image = np.array([normalization[index * length: (index + 1) * length] for index in range(8)])
#     see(image)
#
#
#

