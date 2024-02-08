"""Testing the model for BAyTTA.
===============================================
Version |    Date     |   Author    |   Comment
-----------------------------------------------
0.0     | 8 Feb 2024 | Z. Sherkat | initial version
==============================================="""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import keras
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import statsmodels.api as sm
import BMA
#import load_dataset, metrics

import sys 
sys.path.insert(0, '/src')
from utils import load_dataset_test, metrics



def original_image_evaluation(x, y):
    """Metrics original image."""
    pred_original_image = model.predict(x)
    metrics_orig_imege = metrics.evaluation(pred_original_image, y)
    
    return metrics_orig_imege




def test_time_aug(model, datagen, x, y, tta_steps=6):
    """Metrics Test Time Aug."""
    predictions = []
    preds_tot = []
    prediction = model.predict(x)
    predictions.append(prediction)
    arg = np.argmax(prediction, axis=1)
    preds_tot.append(arg)
    bs = 64
    for i in range(tta_steps):
        prediction = model.predict(datagen.flow(x, batch_size=bs, shuffle=False), steps = len(x)/bs)
        arg = np.argmax(prediction, axis=1)
        preds_tot.append(arg)
        predictions.append(prediction)

    final_pred = np.mean(predictions, axis=0)
    #print('final_pred', final_pred)
    labels = np.argmax(y, axis=1)
    metrics_TTA = metrics.evaluation(final_pred, y)
    UQ_TTA = np.std(metrics_TTA[0])

    return metrics_TTA, preds_tot, labels, UQ_TTA

def BMA_evaluation(metrics_TTA, preds_tot, labels):   
    """Metrics BMA."""                
    pred_tot_df = pd.DataFrame(np.array(preds_tot).T,columns=['aug1', 'aug2', 'aug3', 'aug4', 'aug5', 'aug6', 'aug7'])
    labels_df = pd.DataFrame(np.array(labels).T, columns=['labels'])

    result = BMA.BMA(labels_df, pred_tot_df, RegType = 'Logit', Verbose=False).fit()
    pred_BMA = result.predict(pred_tot_df)
    pred_BMA = pred_BMA > 0.5
    metrics_BMA = metrics.get_metrics(pred_BMA, labels)                
    print('metrics_TTA', metrics_TTA)    
    acc_BMA = metrics_BMA[0]
    acc_TTA = metrics_TTA

    prob = result.uncertainty()
    coff1 = prob.loc[prob['Variable Name'] == 'aug1', 'Probability'].iloc[0]
    coff2 = prob.loc[prob['Variable Name'] == 'aug2', 'Probability'].iloc[0]
    coff3 = prob.loc[prob['Variable Name'] == 'aug3', 'Probability'].iloc[0]
    coff4 = prob.loc[prob['Variable Name'] == 'aug4', 'Probability'].iloc[0]
    coff5 = prob.loc[prob['Variable Name'] == 'aug5', 'Probability'].iloc[0]
    coff6 = prob.loc[prob['Variable Name'] == 'aug6', 'Probability'].iloc[0]
    coff7 = prob.loc[prob['Variable Name'] == 'aug7', 'Probability'].iloc[0]
                                        
    
    UQ_BMA = np.sqrt(((coff1*(acc_TTA[0] - acc_BMA))**2+(coff2*(acc_TTA[1] - acc_BMA))**2+(coff3*(acc_TTA[2] - acc_BMA))**2+(coff4*(acc_TTA[3] - acc_BMA))**2+(coff5*(acc_TTA[4] - acc_BMA))**2+(coff6*(acc_TTA[5] - acc_BMA))**2 + (coff7*(acc_TTA[6] - acc_BMA))**2)/7)*10**2


    return metrics_BMA, UQ_BMA, result

                    
def Log_reg_evaluation(preds_tot, labels):
    """Metrics Logistic."""
    pred_tot_df = pd.DataFrame(np.array(preds_tot).T,columns=['aug1', 'aug2', 'aug3', 'aug4', 'aug5', 'aug6', 'aug7'])
    labels_df = pd.DataFrame(np.array(labels).T, columns=['labels'])

    log_reg = sm.Logit(labels_df, pred_tot_df).fit()
    pred_logit = log_reg.predict(pred_tot_df)  
    metrics_logit = metrics.evaluation(pred_logit, label)                
    

    return metrics_logit                    


def evaluate_aug(model, x, y, tta_steps, aug):
                    
    if args.aug == 'flip':            
        test_datagen_flip = ImageDataGenerator(vertical_flip=True)
        metrics_TTA, preds_tot, labels, UQ_TTA = test_time_aug(model, test_datagen_flip, xtest, ytest, tta_steps=6) 
        metrics_BMA, UQ_BMA, result = BMA_evaluation(metrics_TTA, preds_tot, labels)  
        metrics_logit = Log_reg_evaluation(preds_tot, labels)

    elif args.aug == 'zoom': 
        test_datagen_zoom = ImageDataGenerator(zoom_range=0.1)
        metrics_TTA, preds_tot, labels, UQ_TTA = test_time_aug(model, test_datagen_zoom, xtest, ytest, tta_steps=6)
        metrics_BMA, UQ_BMA, result = BMA_evaluation(metrics_TTA, preds_tot, labels) 
        metrics_logit = Log_reg_evaluation(preds_tot, labels)
                    
    elif args.aug == 'shift': 
        test_datagen_shift = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
        metrics_TTA, preds_tot, labels, UQ_TTA = test_time_aug(model, test_datagen_shift, xtest, ytest, tta_steps=6)
        metrics_BMA, UQ_BMA, result = BMA_evaluation(metrics_TTA, preds_tot, labels)
        metrics_logit = Log_reg_evaluation(preds_tot, labels)            
        
    return  metrics_TTA, metrics_BMA, metrics_logit, UQ_TTA, UQ_BMA, result                
                    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--aug", help="Type pf data augmentation", type=str, default='flip')

    args = parser.parse_args()
    
    mainDataPath = "Path to /test/ dataset"

    test_one = mainDataPath + "/benign/"
    test_two = mainDataPath + "/malignant/"
    
    model_path = "/src/checkpoint/model_SC_Adam.h5"
    model = load_model(model_path) 
                    
    ytest, xtest = load_dataset.load_dataset(test_one, test_two)
                    
    metrics_orig_imege = original_image_evaluation(xtest, ytest) 
    metrics_TTA, metrics_BMA, metrics_logit, UQ_TTA, UQ_BMA, result = evaluate_aug(model, xtest, ytest, tta_steps=6, aug=args.aug)                
    
    print("BMA-summary")
    print(result.summary())


    print('Metrics-accuracy-precision-recall-f1', metrics) 
    print('Metrics-orig_imege-accuracy-precision-recall-f1', metrics_orig_imege)
    print('Metrics-BMA-accuracy-precision-recall-f1', metrics_BMA)
    print('Metrics-logit-accuracy-precision-recall-f1', metrics_logit)


    print('UQ_TTA', )
    print('UQ_BMA', UQ_BMA)              

                    
                    
                    


