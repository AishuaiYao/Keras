from keras.models import *
from keras.layers import *
from keras.optimizers import *



def unet(input_size = (256,256,1)):
    input = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv6)

    conv7 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv8 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv8)

    conv9 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv10 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    up1 = Conv2D(512,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size =(2,2))(conv10))
    merge1 = concatenate([conv8,up1],axis = 3)
    conv11 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    conv12 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

    up2 = Conv2D(256,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size =(2,2))(conv12))
    merge2 = concatenate([conv6,up2],axis = 3)
    conv13 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge2)
    conv14 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv13)

    up3 = Conv2D(128,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size =(2,2))(conv14))
    merge3 = concatenate([conv4,up3],axis = 3)
    conv15 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge3)
    conv16 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv15)

    up4 = Conv2D(64,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size =(2,2))(conv16))
    merge4 = concatenate([conv2,up4],axis = 3)
    conv17 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge4)
    conv18 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv17)
    conv19 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv18)
    conv20 = Conv2D(1, 1, activation='sigmoid')(conv19)

    model = Model(inputs = input ,outputs = conv20)
    model.compile(loss = 'binary_crossentropy',optimizer = Adam(lr =1e-4),metrics = ['accuracy'])

    model.summary()

    return model
































