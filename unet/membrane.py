from tools import *
from unet.model import *
from unet.dataloader import *
from keras.callbacks import ModelCheckpoint

aug_args = dict(rotation = 0.2,
            width_shift_range =0.05,
            height_shift_range = 0.05,
            shear = 0.5,
            zoom = 0.05,
            horizontal_flip =True,
            fill = 'nearest')

train_generator = trainGenerator(2,'./data/membrane/train','image','label')
test_generator = testGenerator('./data/membrane/test')

model = unet()

if os.path.exists('./model/unet.h5'):
    model.load_weights('./model/unet.h5')
model_checkpoint = ModelCheckpoint('./model/unet.h5',monitor ='acc',verbose =1,save_best_only=True)
model.fit_generator(train_generator,steps_per_epoch=300,epochs=10,callbacks =[model_checkpoint])

results = model.predict_generator(test_generator,30,1)
saveResult('./data/membrane/test',results)




