#https://arxiv.org/pdf/0704.1028.pdf
#https://www.kaggle.com/lextoumbourou/blindness-detection-resnet34-ordinal-targets
import pandas as pd
import numpy as np

def classification_output(y_pred):
    '''
    This function gives out categorical(0,1,2,3,4) output varaible by taking in ordinal inputs like \
    [1,1,1,1,0] i.e all the categories before actual category are set to one
    '''
    y_pred_ordinal = np.empty(y_pred.shape, dtype=int)
    y_pred_ordinal[:,4] = y_pred[:,4]

    for i in range(3,-1,-1):
        y_pred_ordinal[:,i] = np.logical_or(y_pred[:,i],y_pred_ordinal[:,i+1])
    
    y_pred = y_pred_ordinal.sum(axis=1)-1
    
    return y_pred