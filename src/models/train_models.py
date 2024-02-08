import os
import sys 
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import keras
from keras import models
from keras import layers
from keras.models import *
from keras.layers import *
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adamax, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications import DenseNet201
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.inception_resnet_v2 import InceptionResNetV2

sys.path.insert(0, 'Path/ to/ src')
from utils import load_dataset, metrics
from constants import constants


def create_model(opts, base_model):
    base_model.trainable=False
    model = Sequential()
    model.add(base_model)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    #model.summary()
    model.compile(optimizer= opts,loss='BinaryCrossentropy',metrics=['accuracy'])
    return model
        

def Optimize(opts,base_model,BS,x_train,y_train,x_test,y_test,epoch=400,verbose=0, history_diagrams=True):
    model = create_model(opts, base_model)
    eary_stopping = EarlyStopping(monitor='val_loss',patience=30)
    callbacks = [eary_stopping]
    train_datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,shear_range = 0.2, zoom_range = 0.2,horizontal_flip=True,vertical_flip=True)
    x = train_datagen.flow(x_train, y_train, batch_size=BS)
    history = model.fit(x, validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // BS, epochs=epoch, callbacks=callbacks)
    #model.save("/src/checkpoint/model_SC_Adam.h5")
    score = model.evaluate(x_test, y_test, verbose=verbose)
    temp_result = "lr: "+str(lr)+"BS:" + str(BS) +"acc: "+str(score)
    if history_diagrams==True:
        fig = plt.figure()
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='test loss')
        plt.title('Learning Curves')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        fig.savefig('loss_'+str(epoch)+' -opt:'+str(opts)+'-dim: '+str(BS)+'.png')
        plt.close()

        fig = plt.figure()
        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='test acc')
        plt.title('Learning Curves')
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.legend()
        fig.savefig('acc_'+str(epoch)+' -opt:'+str(opts)+'-dim: '+str(BS)+'.png')
        plt.close()
    return temp_result    

def train_model(x_train, y_train, x_test, y_test):
                    
    if args.base_models == 'VGG':            
        base_model = VGG16(weights="imagenet", include_top=False,input_shape=(img_size,img_size,3))
        opts=Adam(learning_rate=Learning_Rate_VGG)
        BS = BS_VGG
        Optimize(opts,base_model,BS,x_train,y_train,x_test,y_test,verbose=0)

    elif args.base_models == 'MobileNetV2':            
        base_model = MobileNetV2(weights="imagenet", include_top=False,input_shape=(img_size,img_size,3))
        opts=Adam(learning_rate=Learning_Rate_MobileNetV2)
        BS = BS_MobileNetV2
        Optimize(opts,base_model,BS,x_train,y_train,x_test,y_test)
                    
    elif args.base_models == 'DenseNet201':            
        base_model = DenseNet201(weights="imagenet", include_top=False,input_shape=(img_size,img_size,3))
        opts = Adamax(learning_rate=Learning_Rate_DenseNet201)
        BS = BS_DenseNet201
        Optimize(opts,base_model,BS,x_train,y_train,x_test,y_test)

    elif args.base_models == 'ResNet152V2':            
        base_model = ResNet152V2(weights="imagenet", include_top=False,input_shape=(img_size,img_size,3))
        opts = Adamax(learning_rate=Learning_Rate_ResNet152V2)
        BS = BS_ResNet152V2
        Optimize(opts,base_model,BS,x_train,y_train,x_test,y_test)


    elif args.base_models == 'InceptionResNetV2':            
        base_model = InceptionResNetV2(weights="imagenet", include_top=False,input_shape=(img_size,img_size,3))
        opts = Adamax(learning_rate=Learning_InceptionResNetV2)
        BS = BS_InceptionResNetV2
        Optimize(opts,base_model,BS,x_train,y_train,x_test,y_test)
                      
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--base_models", help="Type pf data augmentation", type=str, default='VGG')

    args = parser.parse_args()
    
    mainDataPath_train = "Path to /train/ dataset"
    mainDataPath_test = "Path to /test/ dataset"

    
    train_one = mainDataPath_train + "/benign/"
    train_two = mainDataPath_train + "/malignant/"
    test_one = mainDataPath_test + "/benign/"
    test_two = mainDataPath_test + "/malignant/"
    
    
    ytrain, xtrain = load_dataset.load_dataset_train(train_one, train_two)
    ytest, xtest = load_dataset.load_dataset_test(train_one, train_two)


    train_model(xtrain, ytrain, xtest, ytest)






