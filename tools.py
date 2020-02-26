import cv2
import os
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

def show_plot(history,path):
    """
    show the train situation
    :param history:the fit return
    :param path:the save dir of graph
    :return: none
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model_accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc='upper left')
    plt.savefig(path+'/accuracy.jpg')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(path+'./loss.jpg')
    plt.legend(['Train','Test'],loc ='upper left')
    plt.show()


def save_heatmap(im_path,model,layer = 1):
    """
    Feature map visualization with unet.Helps you understand semantic segmentation.The default save path is './heatmap'
    :param im_path:test image
    :param layer:the layer of unet which you want to visual
    :return:None
    """
    im = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)/255
    im = cv2.resize(im,(256,256))
    im = np.expand_dims(im,axis = 2)
    im = np.expand_dims(im,axis = 0)

    extractor = K.function([model.layers[0].input],[model.layers[layer].output])# total 36 layers include maxpool upsample and concate
    feature_maps = np.squeeze(extractor([im])[0],axis=0)

    n_channels = int(feature_maps.shape[-1])
    print('total %d maps'%n_channels)
    for i in range(n_channels):
        feature_map = feature_maps[:,:,i]
        plt.imshow(feature_map, cmap = 'jet')
        # plt.colorbar()
        # plt.show()#Use with caution. If you have too many channels, this method can cause a memory explosion. Limit the number of displays before use
        save_dir = './unet/heatmap/%d'%layer
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        feature_map = cv2.resize(feature_map, (256, 256), interpolation=cv2.INTER_NEAREST)
        plt.imsave(save_dir+'/%d.png'%i,feature_map)








