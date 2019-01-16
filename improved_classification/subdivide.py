'''
Code for supervised feature extraction for subtomogram subdivision
'''

import numpy as N

from keras.layers import Input, Dense, Conv3D, Convolution3D, MaxPooling3D, merge, ZeroPadding3D, AveragePooling3D, Dropout, Flatten, Activation, BatchNormalization
from keras.layers.merge import concatenate
import keras.models as KM

## Inception3D
def inception_0(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))

    m = Convolution3D(32, 5, 5, 5, subsample=(1, 1, 1), activation='relu', border_mode='valid', input_shape=())(inputs)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode='same')(m)

    # inception module 0
    branch1x1 = Convolution3D(32, 1, 1, 1, subsample=(1, 1, 1), activation='relu', border_mode='same')(m)
    branch3x3_reduce = Convolution3D(32, 1, 1, 1, subsample=(1, 1, 1), activation='relu', border_mode='same')(m)
    branch3x3 = Convolution3D(64, 3, 3, 3, subsample=(1, 1, 1), activation='relu', border_mode='same')(branch3x3_reduce)
    branch5x5_reduce = Convolution3D(16, 1, 1, 1, subsample=(1, 1, 1), activation='relu', border_mode='same')(m)
    branch5x5 = Convolution3D(32, 5, 5, 5, subsample=(1, 1, 1), activation='relu', border_mode='same')(branch5x5_reduce)
    branch_pool = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), border_mode='same')(m)
    branch_pool_proj = Convolution3D(32, 1, 1, 1, subsample=(1, 1, 1), activation='relu', border_mode='same')(branch_pool)
    m = merge([branch1x1, branch3x3, branch5x5, branch_pool_proj], mode='concat', concat_axis=-1)

    m = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), border_mode='valid')(m)
    m = Flatten()(m)
    m = Dropout(0.7)(m)

    # expliciately seperate Dense and Activation layers in order for projecting to structural feature space
    m = Dense(num_labels, activation='linear')(m)
    m = Activation('softmax')(m)

    mod = KM.Model(input=inputs, output=m)

    return mod
## DSRF3D
def vgg_0(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))

    # modified VGG19 architecture
    bn_axis = 3
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Flatten(name='flatten')(m)
    m = Dense(512, activation='relu', name='fc1')(m)
    m = Dense(512, activation='relu', name='fc2')(m)
    m = Dense(num_labels, activation='softmax')(m)

    mod = KM.Model(input=inputs, output=m)

    return mod

## Chengqian
## VGG 19 modified---Cgg
def Cgg(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))

    # modified VGG19 architecture
    bn_axis = 3

    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)   
 
    m = Flatten(name='flatten')(m)
    m = Dense(1024, activation='relu', name='fc1')(m)
    m = Dropout(0.7)(m)
    
    m = Dense(1024, activation='relu', name='fc2')(m)
    m = Dropout(0.7)(m)   
    m = Dense(num_labels, activation='softmax')(m)

    mod = KM.Model(input=inputs, output=m)

    return mod

## Chengqian Modified Resnet
def Cresnet(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)
   

	
    shortcut = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(m)
   
    #Bottleneck
    mNew = Convolution3D(16, 1, 1, 1, activation='relu', border_mode='same')(m)
    mNew = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(mNew)
    mNew = Convolution3D(16, 1, 1, 1, border_mode='same')(mNew)
    mNew = Dropout(0.7)(mNew)
    mMerge = merge([shortcut, mNew], mode='concat', concat_axis=-1)

    m = Activation('relu')(mMerge)


    shortcut2 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(m)
  

    #Bottleneck
    mNew2 = Convolution3D(16, 1, 1, 1, activation='relu', border_mode='same')(m)
    mNew2 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(mNew2)
    mNew2 = Convolution3D(16, 1, 1, 1, border_mode='same')(mNew2)
    mNew2 = Dropout(0.7)(mNew2)
    mMerge2 = merge([shortcut2, mNew2], mode='concat', concat_axis=-1)
    m = Activation('relu')(mMerge2)


    shortcut3 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(m)
  
     #Bottleneck
    mNew3 = Convolution3D(16, 1, 1, 1, activation='relu', border_mode='same')(m)
    mNew3 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(mNew3)
    mNew3 = Convolution3D(16, 1, 1, 1, border_mode='same')(mNew3)
    mNew3 = Dropout(0.7)(mNew3)
    mMerge3 = merge([shortcut3, mNew3], mode='concat', concat_axis=-1)
    
    m = Activation('relu')(mMerge3)


    shortcut4 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(m)

     #Bottleneck
    mNew4 = Convolution3D(16, 1, 1, 1, activation='relu', border_mode='same')(m)
    mNew4 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(mNew4)
    mNew4 = Convolution3D(16, 1, 1, 1, border_mode='same')(mNew4)
    mNew4 = Dropout(0.7)(mNew4)
    mMerge4 = merge([shortcut4, mNew4], mode='concat', concat_axis=-1)
    m = Activation('relu')(mMerge4)






    m = Flatten(name='flatten')(m)
    m = Dense(1024, activation='relu', name='fc1')(m)
    m = Dropout(0.5)(m)

    m = Dense(1024, activation='relu', name='fc2')(m)
    m = Dropout(0.5)(m)
    m = Dense(num_labels, activation='softmax')(m)


    mod = KM.Model(input=inputs, output=m)

    return mod






def c3d(image_size, num_labels):
    num_channels = 1
    model = Sequential()
    input_shape=(image_size, image_size, image_size, num_channels) # l, h, w, c
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1a',
                            input_shape=input_shape))
    #model.add(Convolution3D(64, 3, 3, 3, activation='relu',
    #                        border_mode='same', name='conv1b'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2a'))
    #model.add(Convolution3D(128, 3, 3, 3, activation='relu',
    #                        border_mode='same', name='conv2b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a'))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b'))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b'))
    
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    
    model.add(Flatten())
    #model.add(BatchNormalization())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(num_labels, activation='softmax', name='fc8'))

    return model


def compile(model):
    import keras.optimizers as KOP
    # 0.003 takes longer and gives about the same accuracy
    kop = KOP.SGD(lr=0.005, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(optimizer=kop, loss='categorical_crossentropy', metrics=['accuracy'])


def train(model, dj, pdb_id_map, nb_epoch):
    dl = list_to_data(dj, pdb_id_map)

    model.fit(dl['data'], dl['labels'], epochs=nb_epoch, shuffle=True, validation_split=validation_split)


def train_validation(model, dj, imagedb, pdb_id_map, nb_epoch, validation_split):
    from sklearn.model_selection import train_test_split
    sp = train_test_split(dj, test_size=validation_split)
    
    train_dl = list_to_data(sp[0],imagedb, pdb_id_map)
    validation_dl = list_to_data(sp[1],imagedb, pdb_id_map)

    model.fit(train_dl['data'], train_dl['labels'], validation_data=(validation_dl['data'], validation_dl['labels']), epochs=nb_epoch, shuffle=True)



def predict(model, dj):

    data = list_to_data(dj)
    pred_prob = model.predict(data)      # predicted probabilities
    pred_labels = pred_prob.argmax(axis=-1)

    return pred_labels



def vol_to_image_stack(vs):
    image_size = vs[0].shape[0]
    sample_size = len(vs)
    num_channels=1

    sample_data = N.zeros((sample_size, image_size, image_size, image_size, num_channels), dtype=N.float32)

    for i,v in enumerate(vs):        sample_data[i, :, :, :, 0] = v

    return sample_data

def pdb_id_label_map(pdb_ids):
    pdb_ids = set(pdb_ids)
    pdb_ids = list(pdb_ids)
    m = {p:i for i,p in enumerate(pdb_ids)}
    return m

def list_to_data(dj,imagedb, pdb_id_map=None):
    re = {}
    vs = [None]*len(dj)
    labels = [None]*len(dj)
    for i,n in enumerate(dj):
	vs[i] = imagedb[n['subtomogram']]
        if pdb_id_map is not None:
            labels[i] = pdb_id_map[n['pdb_id']]
    re['data'] = vol_to_image_stack(vs = vs)

    if pdb_id_map is not None:
       # labels = N.array([pdb_id_map[_['pdb_id']] for _ in dj])
        from keras.utils import np_utils
        labels = np_utils.to_categorical(labels, len(pdb_id_map))
        re['labels'] = labels

    return re




# %reset -f
if __name__ == "__main__":
    import pickle
    from lsm_db import LSM
    dconfig = LSM('/shared/shared/subdivide/data/subtomograms/data_config.db')
    imagedb = LSM('/shared/shared/subdivide/data/subtomograms/image_db.db')
    stat = LSM('/shared/shared/subdivide/data/subtomograms/stat.db')   
    checker = True
    for config in dconfig.keys():
        dj = dconfig[config]['dj']
        pdb_id_map = pdb_id_label_map([_['pdb_id'] for _ in dj]) 
        #model = vgg_0(image_size = stat['op']['size'], num_labels=len(pdb_id_map))


        model = c3d(image_size = stat['op']['size'], num_labels = len(pdb_id_map))
        #if checker == True:
        compile(model)
            #checker = False
        train_validation(model=model, dj=dj,imagedb = imagedb,  pdb_id_map=pdb_id_map, nb_epoch=20, validation_split=0.2)
        
