import keras
from keras.datasets import cifar10
from myNet import Nets

H,W,C = 32,32,3
n_classes = 10
shape = (H,W,C)
input_size = H*W*C
batch_size = 128
epoch=30

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.reshape(-1,H,W,C).astype('float')/255
x_test = x_test.reshape(-1,H,W,C).astype('float')/255
y_train = keras.utils.to_categorical(y_train,n_classes)
y_test = keras.utils.to_categorical(y_test,n_classes)

nets = Nets(n_classes,shape)

model = nets.ResNet18()
model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epoch,batch_size=batch_size)
# model.save('./model/cifar10CNN.h5')
score = model.evaluate(x_test,y_test,batch_size=batch_size)
print(score)






