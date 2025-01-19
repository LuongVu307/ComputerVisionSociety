import numpy as np
import matplotlib.pyplot as plt

from models import Sequential
from optimizers import SGD, Momentum, RMSprop, Adam
from metrics import accuracy, mae, precision, recall
from layers import Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten
from losses import MeanSquaredError, CategoricalCrossEntropy, BinaryCrossEntropy
from activations import ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Linear


def test1():
    model = Sequential()

    X = np.array([  [0.3153062,  0.28318492, 0.24840044],
                    [0.12895783, 0.37114495 ,0.54585694],
                    [0.9598812 , 0.5993223  ,0.62095469],
                    [0.1636091  ,0.14915891 ,0.89525675],
                    [0.61860381 ,0.8417314  ,0.58524656]])
    y = np.array([[0.02117229, 0.01058614],
 [0.02614899, 0.0130745 ],
 [0.05450395, 0.02725198],
 [0.03020062, 0.01510031],
 [0.05113954, 0.02556977]])


    # print(X.shape, y.shape)

    model.add(Dense(5, activation=ReLU()))
    model.add(Dense(6, activation=ReLU()))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation=ReLU()))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=MeanSquaredError())

    y_pred = model.predict(X)
    # print(model.layers[1].W.shape)
#     model.layers[0].W = np.array([[-0.03144386,  0.01393983, -0.02083005,  0.01132874, -0.00293473],
#  [-0.00491062,  0.00833558,  0.00622588, -0.00172509,  0.02183544],
#  [ 0.00754485,  0.01389283, -0.00803183, -0.00196528,  0.00103241]])

#     model.layers[1].W = np.array([[ 0.00246437,  0.00515493],
#  [-0.003632,   -0.01388519],
#  [-0.00454469, -0.01761157],
#  [ 0.00539426,  0.00514456],
#  [-0.00727075, -0.01242132]]
# )

    print(MeanSquaredError().forward(y_pred, y))
    loss1 = MeanSquaredError().forward(model.predict(X), y)
   #  print(model.layers[-1].W[0][0])
    mini = 1e-10
    # print("INITAL W: ", model.layers[-1].W[0][0])
    model.layers[0].W[0][0] += mini
    # print(model.layers[-1].W[0][0])
    loss2 = MeanSquaredError().forward(model.predict(X), y)
    l1 = (loss2-loss1)/mini
    print("TRUE LOSS: ", l1)
    
    model.layers[0].W[0][0] -= mini
   #  print(loss1, loss2)

    # print("AFTER W: ", model.layers[-1].W[0][0])
    # save1 = model.layers[-1].W.copy()
   #  print(MeanSquaredError().forward(model.predict(X), y))
    # print(X.shape)
    model.fit(X, y, epoch=1)
    # save2 = model.layers[-1].W
    l2 = model.layers[0].dW[0][0]
    print("PRED LOSS: ", l2)
    
    print("DIFF: ", l1-l2)
    # print(y_pred.shape, y.shape)
    # y_pred = model.predict(X)
    # print(MeanSquaredError().forward(y_pred, y))
    # print(X.shape)
    print(loss2, MeanSquaredError().forward(model.predict(X), y), sep="\n")

    # print(MeanSquaredError().forward(model.predict(X), y), sep="\n")
    # model.fit(X, y, epoch=1000)
    # print(MeanSquaredError().forward(model.predict(X), y), sep="\n")


def test2():
    model = Sequential()


    X = np.array([np.random.randint(0, 256, size=(10, 10, 2))])/255
    y = np.array([-0.1])
    # plt.imshow(X, cmap="Greys")
    # plt.show()
    # print(X)

    model.add(Conv2D(5, 3, 2, activation=ReLU()))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation=Sigmoid()))

    model.compile(loss=BinaryCrossEntropy(), optimizer=SGD())

    y_pred = model.predict(X)
    
    
    print(BinaryCrossEntropy().forward(y_pred, y))
    loss1 = BinaryCrossEntropy().forward(model.predict(X), y)
   #  print(model.layers[-1].W[0][0])
    mini = 1e-10
    # print("INITAL W: ", model.layers[-1].W[0][0])
    model.layers[0].W[0][0][0][0] += mini
    # print(model.layers[-1].W[0][0])
    loss2 = BinaryCrossEntropy().forward(model.predict(X), y)
    l1 = (loss2-loss1)/mini
    print("TRUE LOSS: ", l1)
    
    model.layers[0].W[0][0][0][0] -= mini

    model.fit(X, y, epoch=1)
    # save2 = model.layers[-1].W
    l2 = model.layers[0].dW[0][0][0][0]
    print("PRED LOSS: ", l2)
    
    print("DIFF: ", l1-l2)
    # print(y_pred.shape, y.shape)
    # y_pred = model.predict(X)
    # print(MeanSquaredError().forward(y_pred, y))

    print(loss2, BinaryCrossEntropy().forward(model.predict(X), y), sep="\n")


import tensorflow as tf
def test3():
    

    (x_train, y_train), (x_test, y_test)= tf.keras.datasets.mnist.load_data()

    X_train, X_test, Y_train, Y_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3), np.expand_dims(y_train, axis=1), np.expand_dims(y_test, axis=1)

    X_train, X_test, Y_train, Y_test = X_train[:2000], X_test[:2000], Y_train[:1000], Y_test[:1000]
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


    model1 = Sequential()

    model1.add(Conv2D(64, 3, activation=ReLU()))
    model1.add(MaxPool2D((2, 2)))

    model1.add(Flatten())

    model1.add(Dense(32, ReLU()))
    model1.add(Dropout(0.5))
    model1.add(Dense(1, Sigmoid()))

    model1.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(), metrics=[accuracy])

    model1.fit(X_train, Y_train, epoch=10, batch_size=64)



test3()