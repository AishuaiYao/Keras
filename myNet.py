from keras.models import Sequential,Model
from keras.layers import Input,Flatten,Dropout,regularizers,Add
from keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,MaxPooling2D,Dense
from keras.layers import ZeroPadding2D,AveragePooling2D
from keras.optimizers import SGD

class Nets():
    def __init__(self,n_classes,shape = None,fc_size = None):
        self.n_classes = n_classes
        self.shape = shape
        self.fc_size = fc_size

    def MLP(self):
        model = Sequential()
        model.add(Dense(self.fc_size,input_dim = self.fc_size,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes,activation='softmax'))

        model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        return model

    def CNN(self):
        """
        随意搭建的普通CNN，mnist直接使用时过拟合现象严重，加入了Dropout和l2正则缓解。
        另外使用SGD时经常出现更新停滞，可能是陷入了局部极小值，Adam比较稳定，每次都能更新
        :return: model
        """
        model = Sequential()
        model.add(Conv2D(16,(3,3),activation='relu',input_shape = self.shape))
        model.add(Conv2D(32,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128,(3,3),kernel_regularizer=regularizers.l2(0.1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes,activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics = ['accuracy'])
        return model


    def VGG(self):
        """
        由于使用自己的笔记本显卡1050-4G显存实验，所以只能跑得动VGG11，层数再多就OOM了。实验中由于网络过深出现了梯度
        消失，导致损失不在下降，所以每层后面加入了BN操作后解决。
        :return: VGG11模型
        """
        model = Sequential()
        model.add(Conv2D(64, (3, 3),input_shape=self.shape))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2),strides=(2,2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2),strides=(2,2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2),strides=(2,2)))

        model.add(Conv2D(512, (3, 3)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2),strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(1024,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        return model


    def identity_block(self,x,filters):
        shortcut = x
        f1,f2,f3 = filters

        x = Conv2D(f1,(1,1),padding='valid')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(f2,(3,3),padding='same')(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = Conv2D(f3,(1,1),padding='valid')(x)
        x = BatchNormalization(axis=3)(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        return x


    def convolutional_block(self,x,filters,stride):
        shortcut = x
        f1,f2,f3 = filters

        x = Conv2D(f1,(1,1),padding='valid',strides=stride)(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = Conv2D(f2, (3,3), padding='same',strides =1)(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(f3, (1, 1), padding='valid',strides = 1)(x)
        x = BatchNormalization(axis=3)(x)
        shortcut = Conv2D(f3,(1,1),padding = 'valid',strides = stride)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)

        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        return x


    def basic_block(self,x,filters,stride,name):
        shortcut = x

        x = Conv2D(filters,(3,3),strides=stride,padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, (3, 3), strides=1,padding='same' )(x)
        x = BatchNormalization(axis=3)(x)

        if x.shape != shortcut.shape:
            shortcut = Conv2D(filters,(1,1),strides = stride,name=name)(shortcut)
            shortcut = BatchNormalization(axis=3)(shortcut)

        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        return x


    def ResNet18(self):
        """
        还是太深的话，笔记本带不动，ResNet18还可以，该模型比较稳定，参数量小，没有梯度消失现象
        :return:ResNet18模型
        """
        input = Input(self.shape)

        x = ZeroPadding2D((3,3))(input)
        x = Conv2D(64,(7,7),strides=2)(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3,3),strides = (2,2),padding='same')(x)

        x = self.basic_block(x,64,1,name='shortcut1')
        x = self.basic_block(x,64,1,name='shortcut2')

        x = self.basic_block(x, 128, 2,name='shortcut3')
        x = self.basic_block(x, 128, 1,name='shortcut4')

        x = self.basic_block(x, 256, 2,name='shortcut5')
        x = self.basic_block(x, 256, 1,name='shortcut6')

        x = self.basic_block(x, 512, 2,name='shortcut7')
        x = self.basic_block(x, 512, 1,name='shortcut8')

        size = int(x.shape[1])
        x = AveragePooling2D(pool_size=(size,size))(x)

        x = Flatten()(x)
        x = Dense(self.n_classes,activation='softmax')(x)

        model = Model(inputs = input,outputs=x)
        model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

        return model

    def ResNet50(self):
        """
        注释了好多后笔记本才带的动
        :return: ResNet50模型
        """
        input = Input(self.shape)

        x = ZeroPadding2D((3,3))(input)
        x = Conv2D(64,(7,7),strides=(2,2))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size =(3,3),strides=(2,2),padding='same')(x)

        x = self.convolutional_block(x,[64,64,256],stride=1)
        x = self.identity_block(x,[64,64,256])
        # x = self.identity_block(x, [64, 64, 256])

        x = self.convolutional_block(x,[128,128,512],stride=1)
        x = self.identity_block(x,[128,128,512])
        # x = self.identity_block(x, [128, 128, 512])
        # x = self.identity_block(x, [128, 128, 512])
        #
        x = self.convolutional_block(x,[256,256,1024],stride=2)
        x = self.identity_block(x,[256,256,1024])
        # x = self.identity_block(x, [256, 256, 1024])
        # x = self.identity_block(x, [256, 256, 1024])
        # x = self.identity_block(x, [256, 256, 1024])

        x = self.convolutional_block(x,[512,512,2048],stride=2 )
        x = self.identity_block(x,[512,512,2048])
        # x = self.identity_block(x, [512, 512, 2048])

        size = int(x.shape[1])
        x = AveragePooling2D(pool_size=(size, size))(x)

        x = Flatten()(x)
        x = Dense(self.n_classes,activation='softmax')(x)

        model = Model(inputs = input,outputs= x)
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        return model





































