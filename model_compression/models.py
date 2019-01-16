'''
Code for model structures and data processing functions
'''

import numpy as N
import pickle
from keras.layers import Input, Dense, Conv3D, Convolution3D, MaxPooling3D, merge, ZeroPadding3D, AveragePooling3D, Dropout, Flatten, Activation, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model
import keras.models as KM
from keras.metrics import categorical_accuracy

## VGG 19 modified---Cgg
def DSRF3D_v2(image_size, num_labels):
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
    m = Dense(num_labels)(m)
    m = Activation('softmax')(m)

    mod = KM.Model(input=inputs, output=m)

    return mod


def student3(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))

    # modified VGG19 architecture
    bn_axis = 3
    m = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    m = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(3,3,3), strides=(3,3,3))(m)

    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3))(m)


    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3))(m)   
    m = Flatten(name='flatten')(m)
    m = Dense(1024,activation='relu', name='fc1')(m)
    #m = Dropout(0.2)(m)

    #m = Dense(1024,activation='relu', name='fc2')(m)
    #m = Dropout(0.5)(m)
    m = Dense(num_labels)(m)
    #m = Activation('softmax')(m)

    mod = KM.Model(input=inputs, output=m)

    return mod

def student2(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))

    # modified VGG19 architecture
    bn_axis = 3
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(3,3,3), strides=(3,3,3))(m)

    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3))(m)


    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3))(m)   
    m = Flatten(name='flatten')(m)
    m = Dense(1024,activation='relu', name='fc1')(m)
    #m = Dropout(0.2)(m)

    #m = Dense(1024,activation='relu', name='fc2')(m)
    #m = Dropout(0.5)(m)
    m = Dense(num_labels)(m)
    #m = Activation('softmax')(m)

    mod = KM.Model(input=inputs, output=m)

    return mod

def student1(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))

    # modified VGG19 architecture
    bn_axis = 3
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(3,3,3), strides=(3,3,3))(m)

    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3))(m)


    m = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(m)
    m = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3))(m)   
    m = Flatten(name='flatten')(m)
    m = Dense(1024,activation='relu', name='fc1')(m)
    #m = Dropout(0.2)(m)

    #m = Dense(1024,activation='relu', name='fc2')(m)
    #m = Dropout(0.5)(m)
    m = Dense(num_labels)(m)
    #m = Activation('softmax')(m)

    mod = KM.Model(input=inputs, output=m)

    return mod


def compile(model):
    import keras.optimizers as KOP
    # 0.003 takes longer and gives about the same accuracy
    kop = KOP.SGD(lr=0.005, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(optimizer=kop, loss='mean_squared_error', metrics=[accuracy])


def train(model, dj, pdb_id_map, nb_epoch):
    dl = list_to_data(dj, pdb_id_map)

    model.fit(dl['data'], dl['labels'], epochs=nb_epoch, shuffle=True, validation_split=validation_split)


def train_validation(model, dj, imagedb, pdb_id_map, nb_epoch, validation_split, dataNumber):

    from keras.callbacks import EarlyStopping
    from keras.utils import np_utils

    with open("trainfolder/train"+str(dataNumber)+".txt","rb") as fp1:
        train_pure = pickle.load(fp1)
    with open("valfolder/val"+str(dataNumber)+".txt","rb") as fp2:
        validationsets = pickle.load(fp2)
    with open("testfolder/test"+str(dataNumber)+".txt","rb") as fp3:
        testsets = pickle.load(fp3)
    

    train_dl = list_to_data(train_pure,imagedb, pdb_id_map)
    validation_dl = list_to_data(validationsets, imagedb, pdb_id_map)
    test_dl = list_to_data(testsets, imagedb, pdb_id_map)
   
    callbacks = [EarlyStopping(monitor='val_loss',patience=5,verbose=0)]
    model.fit(train_dl['data'], train_dl['labels'],validation_data=(validation_dl['data'],validation_dl['labels']), epochs=nb_epoch, shuffle=True, callbacks = callbacks)
    model.save("teacher_model.h5")
    scores = model.evaluate(test_dl['data'], test_dl['labels'])
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save("org_model.h5")

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
    t = 1
    vs = [None]*len(dj)
    labels = [None]*len(dj)
    for i,n in enumerate(dj):
	vs[i] = imagedb[n['subtomogram']]
        t = t + 1
        if pdb_id_map is not None:
            labels[i] = pdb_id_map[n['pdb_id']]
    re['data'] = vol_to_image_stack(vs = vs)

    if pdb_id_map is not None:
        labels = N.array([pdb_id_map[_['pdb_id']] for _ in dj])
        from keras.utils import np_utils
        labels = np_utils.to_categorical(labels, len(pdb_id_map))
    re['labels'] = labels

    return re


def accuracy(y_true, y_pred):
    prob_tea=Activation('softmax')(y_true)
    prob_stu=Activation('softmax')(y_pred)
    return categorical_accuracy(prob_tea, prob_stu)


