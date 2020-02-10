import cv2
import numpy as np
import random
import keras
import os

class Dataloader():
    def __init__(self,path,n_classes = None):
        self.path = path
        self.files = os.listdir(self.path)
        self.n_classes = n_classes


    def load_data(self,name):
        """
        负责将数据一张一张读入并根据文件名（例23_D）生成标签，23只是数据的序号，根据
        最后的字母D打标签
        :param name:数据的路径
        :return:图像数据和标签
        """
        im = cv2.imread(name)
        im = cv2.resize(im,(224,224))
        label = name.split('_')[-1][0]
        if label == 'A':
            label = 0
        elif label == 'B':
            label = 1
        elif label == 'C':
            label = 2
        elif label == 'D':
            label = 3
        elif label == 'E':
            label = 4
        im = np.array(im).astype('float')/255
        label = keras.utils.to_categorical(label,self.n_classes)
        return im,label


    def load_predict_data(self,name,isgray= False,input_size=(28,28)):
        """
        测试模型的时候使用，一次测一张
        :param name: 文件名和路径
        :param isgray:如果像mnist一类，输入数据是灰度图赋值true
        :param input_size:传入图像的长宽
        :return: 格式如（1,28,28,1）的数据
        """
        im = cv2.imread(name)
        im = cv2.resize(im,input_size)
        if isgray:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = np.expand_dims(im, axis=2)

        cv2.imshow('im_show', im)
        cv2.waitKey(1000)
        im = np.expand_dims(im, axis=0)
        im = np.array(im).astype('float') / 255
        return im


    def traverse_each_folder(self):
        folders = sorted(os.listdir(self.path))
        sign = 0
        for folder in folders:
            files = os.listdir(self.path+'/'+folder)
            for i in range(len(files)):
                im = cv2.imread(self.path+'/'+folder+'/'+files[i])
                im = cv2.resize(im,(224,224))
            sign+=1
        return np.array(im),sign


    def shuffle(self,path):
        files = os.listdir(path)
        random.shuffle(files)
        cnt=0
        for file in self.files:
            if file[0]=='A':
                os.rename(self.path+'/'+file,self.path+'/'+str(cnt)+'_A.jpg')
            elif file[0]=='B':
                os.rename(self.path+'/'+file,self.path+'/'+str(cnt)+'_B.jpg')
            elif file[0]=='C':
                os.rename(self.path+'/'+file,self.path+'/'+str(cnt)+'_C.jpg')
            elif file[0]=='D':
                os.rename(self.path+'/'+file,self.path+'/'+str(cnt)+'_D.jpg')
            elif file[0]=='E':
                os.rename(self.path+'/'+file,self.path+'/'+str(cnt)+'_E.jpg')
            cnt+=1

class DataGenerator(Dataloader):
    """
    继承自Dataloader的迭代器类，如果训练的数据量比较大，就不能在将数据一次性全部读入内存。
    keras这个时候就需要使用python的生成器来分批次训练数据。此类只有train_generator的是迭代
    器，因为训练数据最多。valid_generator只返回数据和标签的元组
    """
    def __init__(self,path,n_classes):
        Dataloader.__init__(self,path,n_classes)


    def train_generator(self,batch_size):
        X = []
        Y = []
        cnt = 0
        while 1:
            for file in self.files:
                data,label = self.load_data(os.path.join(self.path,file))
                X.append(data)
                Y.append(label)
                cnt+=1
                if cnt==batch_size:
                    cnt=0
                    yield (np.array(X),np.squeeze(np.array(Y)))
                    X = []
                    Y = []


    def valid_generator(self):
        X = []
        Y = []
        for file in self.files:
            data, label = self.load_data(os.path.join(self.path, file))
            X.append(data)
            Y.append(label)
        X = np.array(X)
        Y = np.squeeze(np.array(Y))
        return (X,Y)



