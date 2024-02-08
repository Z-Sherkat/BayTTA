import os
import numpy as np
from PIL import Image 
from tensorflow.keras.utils import to_categorical

img_size = 224
num_classes = 2 

#######################################
#       Label train data
#######################################
W_train = []
M_train = []
label = []

def load_dataset_train(train_W, train_M):
    for i in os.listdir(train_W): 
        if os.path.isfile(train_W + i):        
            image_W = np.array(Image.open(train_W + i))
            image_W = np.resize(image_W,(img_size,img_size,3))
            image_W = image_W.astype('float32')
            image_W /= 255  
            W_train.append(image_W)
            label.append(1)
            
    for j in os.listdir(train_M): 
        if os.path.isfile(train_M + j): # check image in file
            image_M = np.array(Image.open(train_M + j))
            image_M = np.resize(image_M,(img_size,img_size,3))
            image_M = image_M.astype('float32')
            image_M /= 255  
            M_train.append(image_M)
            label.append(0)
            


    X_train = np.concatenate((W_train,M_train),axis=0)
    Y_train = np.asarray(label) 
    Y_train= Y_train.reshape(Y_train.shape[0],1)


    print("Women:",np.shape(W_train) , "men:",np.shape(M_train))
    print("train_dataset:",np.shape(X_train), "train_values:",np.shape(Y_train))

    ytrain = to_categorical(Y_train,num_classes)
    xtrain = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 3)

    return ytrain, xtrain

#######################################
#         Label test data
#######################################
W_test = []
M_test = []
label_test = []
def load_dataset_test(test_W, test_M):
    for i in os.listdir(test_W): 
        #print(i)
        #print(test_W + i)
        if os.path.isfile(test_W + i):        
            image_W = np.array(Image.open(test_W + i))
            image_W = np.resize(image_W,(img_size,img_size,3))
            image_W = image_W.astype('float32')
            image_W /= 255  
            W_test.append(image_W)
            label_test.append(1)

    for j in os.listdir(test_M): 
        if os.path.isfile(test_M + j): # check image in file
            image_M = np.array(Image.open(test_M + j))
            image_M = np.resize(image_M,(img_size,img_size,3))
            image_M = image_M.astype('float32')
            image_M /= 255  
            M_test.append(image_M)
            label_test.append(0)



    X_test = np.concatenate((W_test,M_test),axis=0)
    Y_test = np.asarray(label_test) 
    Y_test= Y_test.reshape(Y_test.shape[0],1)


    print("1:",np.shape(W_test) , "2:",np.shape(M_test))
    print("test_x:",np.shape(X_test), "test_y:",np.shape(Y_test))
    ytest = to_categorical(Y_test,num_classes)
    xtest = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 3)
    
    return ytest, xtest
