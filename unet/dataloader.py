import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator


Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask):
    mask[mask>0.5] =1
    mask[mask<=0.5] = 0
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,img_mode = 'grayscale',mask_mode='grayscale'
                   ,image_prefix ='image',mask_prefix ='mask',save_dir = None,target_size =(256,256),seed=1):
    idg = ImageDataGenerator(rescale = 1/255.0)
    mdg = ImageDataGenerator(rescale = 1/255.0)
    image_generator = idg.flow_from_directory(
        directory = train_path,
        color_mode = img_mode,
        classes = [image_folder],
        class_mode = None,
        target_size = target_size,
        batch_size=batch_size,
        save_to_dir=save_dir,
        save_prefix = image_prefix,
        seed = seed
    )
    mask_generator = mdg.flow_from_directory(
        directory=train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode=mask_mode,
        target_size = target_size,
        batch_size=batch_size,
        save_to_dir=save_dir,
        save_prefix = mask_prefix,
        seed =seed
    )
    train_generator = zip(image_generator,mask_generator)
    for img,mask in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)


def testGenerator(test_path,num_image = 30,flag_multi_classes=False,target_size=(256,256),gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,'%d.png'%i),as_gray=gray)/255.0
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if not flag_multi_classes else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def labelVisualize(n_class,color_dict,im):
    im = im[:,:,0] if len(im.shape) == 3 else im
    result = np.zeros(im.shape+(3,))
    for i in range(n_class):
        result[im == i,:] = color_dict[i]
    return result/255.0


def saveResult(path,npyfile,flag_multi_class = False,n_class =2):
    for i,file in enumerate(npyfile):
        im = labelVisualize(n_class,COLOR_DICT,file) if flag_multi_class else file[:,:,0]
        io.imsave(os.path.join(path,'%d_predict.png'%i),im)






















