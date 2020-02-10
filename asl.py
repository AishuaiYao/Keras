from dataloader import DataGenerator,Dataloader
from myNet import Nets
import numpy as np
import os
import time

H,W,C = 224,224,3
shape = (H,W,C)
n_classes = 5
batch_size = 16
epochs = 10
save_model = './model/aslResNet18.h5'

train_data = DataGenerator('./train',n_classes)
valid_data = DataGenerator('./valid',n_classes).valid_generator()

nets = Nets(n_classes,shape)
model = nets.VGG()

def train():
    model.summary()
    #fit_generator需要使用迭代器
    model.fit_generator(train_data.train_generator(batch_size),epochs=epochs,validation_data= valid_data,steps_per_epoch=len(train_data.files)//batch_size)
    # model.save_weights(save_model)

def test(path):
    model.load_weights(save_model)
    dataloader = Dataloader(path)
    files = (os.listdir(path))
    for file in files:
        test_data = dataloader.load_predict_data(os.path.join(path, file), input_size=(H, W))
        time_start = time.time()
        result = model.predict(test_data)
        time_end = time.time()
        pred = np.argmax(result, axis=1)
        print('classes : %s \t cost : %04f s'%(pred,time_end-time_start))


if __name__ =='__main__':
    train()
    data_path= './test/asltest'
    # test(data_path)







