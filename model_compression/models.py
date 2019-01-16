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


def test_model(image_size, num_labels):
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

## Chengqian
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

## Chengqian Modified Resnet
def RB3D(image_size, num_labels):
    num_channels=1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)
   

	
    shortcut = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(m)
   
    #Bottleneck
    mNew = Convolution3D(16, 1, 1, 1, activation='relu', border_mode='same')(m)
    mNew = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(mNew)
    mNew = Convolution3D(16, 1, 1, 1, border_mode='same')(mNew)
    mMerge = merge([shortcut, mNew], mode='concat', concat_axis=-1)

    m = Activation('relu')(mMerge)


    shortcut2 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(m)
  

    #Bottleneck
    mNew2 = Convolution3D(16, 1, 1, 1, activation='relu', border_mode='same')(m)
    mNew2 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(mNew2)
    mNew2 = Convolution3D(16, 1, 1, 1, border_mode='same')(mNew2)
    mMerge2 = merge([shortcut2, mNew2], mode='concat', concat_axis=-1)
    m = Activation('relu')(mMerge2)


    shortcut3 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(m)
  
     #Bottleneck
    mNew3 = Convolution3D(16, 1, 1, 1, activation='relu', border_mode='same')(m)
    mNew3 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(mNew3)
    mNew3 = Convolution3D(16, 1, 1, 1, border_mode='same')(mNew3)
    mMerge3 = merge([shortcut3, mNew3], mode='concat', concat_axis=-1)
    
    m = Activation('relu')(mMerge3)


    shortcut4 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(m)

     #Bottleneck
    mNew4 = Convolution3D(16, 1, 1, 1, activation='relu', border_mode='same')(m)
    mNew4 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(mNew4)
    mNew4 = Convolution3D(16, 1, 1, 1, border_mode='same')(mNew4)
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






def CB3D(image_size, num_labels):
    num_channels = 1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels))

    model = Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same')(inputs)
    model = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)

    # 2nd layer group

    model = Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same')(model)
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(model)

   
    # 3rd layer group

    model = Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same')(model)
    model = Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same')(model)
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(model)



    # 4th layer group

    model = Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same')(model)
    model = Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same')(model)
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(model)



    # 5th layer group

    model = Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same')(model)
    model = Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same')(model)
    model = ZeroPadding3D(padding = (0, 1, 1))(model)
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(model)
    model = Flatten(name='flatten')(model)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dropout(0.5)(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dropout(0.5)(model)

    model = Dense(num_labels, activation='softmax')(model)
    mod = KM.Model(input=inputs, output=model)

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
#Spliting the data into different sets for training, validation and testing
#    from sklearn.model_selection import train_test_split
#    sp = train_test_split(dj, test_size = validation_split)
#    trainsets = sp[0]
#    testsets = sp[1]
#    trainsp = train_test_split(trainsets, test_size = validation_split)
#    train_pure = trainsp[0]
#Writing the datasets
#    validationsets = trainsp[1]
#    with open("trainfolder/train"+str(dataNumber)+".txt","a") as fp1:
#	pickle.dump(train_pure,fp1)
#    with open("valfolder/val"+str(dataNumber)+".txt","a") as fp2:
#        pickle.dump(validationsets,fp2)
#    with open("testfolder/test"+str(dataNumber)+".txt","a") as fp3:
#        pickle.dump(testsets,fp3)
#Loading the datasets
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

# %reset -f

if __name__ == "__main__":
    import pickle
    from lsm_db import LSM
    dconfig = LSM('/shared/shared/subdivide/data/subtomograms/data_config.db')
    imagedb = LSM('/shared/shared/subdivide/data/subtomograms/image_db.db')
    stat = LSM('/shared/shared/subdivide/data/subtomograms/stat.db')
    dataNumber = 1
    for config in dconfig.keys():
        if dataNumber == 9:
            dj=dconfig[config]['dj']
            pdb_id_map=pdb_id_label_map([_['pdb_id'] for _ in dj])
            from keras.models import load_model
            logits_model=load_model("org_model.h5")
            from keras.utils import np_utils
            from keras.callbacks import EarlyStopping
            with open("trainfolder/train"+str(dataNumber)+".txt","rb") as fp1:
                train_pure = pickle.load(fp1)
            with open("valfolder/val"+str(dataNumber)+".txt","rb") as fp2:
                validationsets = pickle.load(fp2)
            with open("testfolder/test"+str(dataNumber)+".txt","rb") as fp3:
                testsets = pickle.load(fp3)
            train_dl = list_to_data(train_pure,imagedb, pdb_id_map)
            validation_dl = list_to_data(validationsets, imagedb, pdb_id_map)
            test_dl=list_to_data(testsets,imagedb,pdb_id_map)
            #logits_model.summary()
            #logits_model.layers.pop()
            #logits_model=Model(logits_model.input,logits_model.layers[-1].output)
            #logits_model.summary()
            #softmax_scores=softmax_model.evaluate(test_dl['data'],test_dl['labels'])
            #logits_scores=logits_model.evaluate(train_dl['data'],train_dl['labels'])
            #print("\n%s: %.2f%%" %(logits_model.metrics_names[1],logits_scores[1]*100))
            #print("\n%s: %.2f%%" %(softmax_model.metrics_names[1],softmax_scores[1]*100))
            #print softmax_model.predict(test_dl['data'])[0][0:20]
            #print logits_model.predict(train_dl['data'])[0:10][0:22]
            #print logits_model.predict(train_dl['data']).shape

            #train_logits=logits_model.predict(train_dl['data'])
            #val_logits=logits_model.predict(validation_dl['data'])
            #N.save('train_logits.npy',train_logits)
            #N.save('val_logits.npy',val_logits)

            train_logits=N.load('train_logits.npy')
            val_logits=N.load('val_logits.npy')
            print train_dl['labels'].shape
            print train_logits.shape
            model=DSRF3D_v2(image_size=stat['op']['size'],num_labels=len(pdb_id_map))
            model.layers.pop()
            compile(model)
            model.summary()
            callbacks = [EarlyStopping(monitor='val_loss',patience=5,verbose=0)]
            model.fit(train_dl['data'], train_logits,validation_data=(validation_dl['data'],val_logits), epochs=20, shuffle=True, callbacks = callbacks)
            model.save("trained_logits_model.h5")
            #print(test_dl['data'].shape)
            #print(test_dl['labels'].shape)
        dataNumber=dataNumber+1

