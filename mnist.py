import keras
import cv2
from keras.datasets import mnist
from myNet import Nets
from dataloader import Dataloader

learning_rate = 0.001
training_iters = 20
batch_size = 128
display_step = 10

H, W, C = 28, 28, 1
input_size = H*W
shape = (H,W,C)
n_classes = 10

nets = Nets(n_classes,shape)
model = nets.CNN()
datasets = Dataloader('./test/mnisttest')

def train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # cv2.imshow('im',x_train[0])
    # cv2.waitKey(1000)
    x_train = x_train.reshape(-1, H, W, 1).astype('float') / 255
    x_test = x_test.reshape(-1, H, W, 1).astype('float') / 255
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    model.summary()
    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size = batch_size)
    # model.save('./model/mnistCNN.h5')
    scores = model.evaluate(x_test,y_test,batch_size=batch_size)
    print(scores)


def test():
    model.load_weights('./model/mnistCNN.h5')
    import os
    import numpy as np
    files = os.listdir(datasets.path)
    for file in files:
        x = datasets.load_predict_data(os.path.join(datasets.path,file),isgray=True)
        y_pred = model.predict(x)  #
        print(np.argmax(y_pred))


if __name__=='__main__':
    train()
    # test()


