from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer

from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import DMatrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from scipy import stats as st
from typing import Tuple
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from decimal import *
import json
from csv import DictWriter
from collections import Counter
import sys
import copy 
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

plt.rcParams.update({'font.size': 22})

# columns over which it makes more sense to sum the metric instead of averaging 
# This is needed because the sampling rate of the logs is not periodic so having larger number 
# of samples but with no macSdu (0) can cause the average to be skewed   
sum_cols = ['macSduInBytes', 'tbSizeInBits', ]

just_for_reference_cols = ['load_Mbps', 'position', 'owd_dl', 'owd_ul' ,'timeStamp']

non_gen_cols = [
             # This is a blacklist of columns that I don't want it to use because domain knowledge indicates that it will result in a model that will not generalize
             # I have basically tried to include here any feature  that has the wod index in it or is related to an index, except mcsIndex which is a different thing 
             'dl_dlCcIndex', 'dl_loadUe_dlCcIndex', 
             'ul_beamIndex', 'dl_beamIndex', 'dl_loadUe_beamIndex', 'ul_loadUe_beamIndex',  
             'wbm_loadUe_WBeamIndexNewBest', 'wbm_loadUe_WBeamIndexCurrent', 'wbm_WBeamIndexNewBest',  'wbm_WBeamIndexCurrent',
             'nbm_WBeamIndex', 'nbm_loadUe_WBeamIndex', 'nbm_NBeamIndexChosen', 'nbm_loadUe_NBeamIndexChosen', 'nbm_NBeamIndexCurrent', 'nbm_loadUe_NBeamIndexCurrent', 
             'nbm_beam01', 'nbm_beam02', 'nbm_beam03', 'nbm_beam11', 'nbm_beam12', 'nbm_beam13',
             'nbm_loadUe_beam01', 'nbm_loadUe_beam02', 'nbm_loadUe_beam03', 'nbm_loadUe_beam11', 'nbm_loadUe_beam12', 'nbm_loadUe_beam13',
             'wbm_beam1', 'wbm_beam2', 'wbm_beam3', 'wbm_beam4',
             'wbm_loadUe_beam1', 'wbm_loadUe_beam2', 'wbm_loadUe_beam3', 'wbm_loadUe_beam4'
            ]

const_val_cols = ['ul_startPrb', 'ul_isTransformPrecoding', 'ul_BSValueLcg0', 'ul_BSValueLcg1',
 'ul_BSValueLcg2', 'ul_BSValueLcg3', 'ul_BSValueLcg4', 'ul_BSValueLcg6',
 'ul_BSValueLcg7', 'ul_BSRestimateLcg0', 'ul_BSRestimateLcg1',
 'ul_BSRestimateLcg2', 'ul_BSRestimateLcg3', 'ul_BSRestimateLcg4',
 'ul_BSRestimateLcg6', 'ul_BSRestimateLcg7', 'ul_VBit',
 'ul_wbUsedNbOverridden', 'ul_carrierAggregationUsed',
 'ul_clpcCarrierDemand', 'ul_isClpcSaturated', 'ul_totalCellsReqScheduling',
 'ul_linkAdaptationUeMode', 'ul_isDrxEnabled', 
 #--------
 'dl_sectorIndex',
 'dl_bbBearerRef1', 'dl_drb1data', 'dl_drb1input', 'dl_bbBearerRef2',
 'dl_drb2data', 'dl_drb2input', 'dl_bbBearerRef3', 'dl_drb3data',
 'dl_drb3input', 'dl_bbBearerRef4', 'dl_drb4data', 'dl_drb4input',
 'dl_bbBearerRef5', 'dl_drb5data', 'dl_drb5input', 'dl_bbBearerRef6',
 'dl_drb6data', 'dl_drb6input', 'dl_bbBearerRef7', 'dl_drb7data',
 'dl_drb7input', 'dl_wbUsedNbOverridden', 'dl_pucchFormatType',
 'dl_linkAdaptationUeMode', 'dl_isDrxEnabled', 
 #--------
 'nbm_rsrp02', 'nbm_rsrp03',
 'nbm_rsrp12', 'nbm_rsrp13', 
 #--------
 'ul_loadUe_startPrb',
 'ul_loadUe_isTransformPrecoding', 'ul_loadUe_BSValueLcg0',
 'ul_loadUe_BSValueLcg1', 'ul_loadUe_BSValueLcg2', 'ul_loadUe_BSValueLcg3',
 'ul_loadUe_BSValueLcg4', 'ul_loadUe_BSValueLcg6', 'ul_loadUe_BSValueLcg7',
 'ul_loadUe_BSRestimateLcg0', 'ul_loadUe_BSRestimateLcg1',
 'ul_loadUe_BSRestimateLcg2', 'ul_loadUe_BSRestimateLcg3',
 'ul_loadUe_BSRestimateLcg4', 'ul_loadUe_BSRestimateLcg6',
 'ul_loadUe_BSRestimateLcg7', 'ul_loadUe_VBit',
 'ul_loadUe_wbUsedNbOverridden', 'ul_loadUe_carrierAggregationUsed',
 'ul_loadUe_clpcCarrierDemand', 'ul_loadUe_isClpcSaturated',
 'ul_loadUe_totalCellsReqScheduling', 'ul_loadUe_linkAdaptationUeMode',
 'ul_loadUe_isDrxEnabled', 
 #--------
 'dl_loadUe_sectorIndex', 'dl_loadUe_bbBearerRef1',
 'dl_loadUe_drb1data', 'dl_loadUe_drb1input', 'dl_loadUe_bbBearerRef2',
 'dl_loadUe_drb2data', 'dl_loadUe_drb2input', 'dl_loadUe_bbBearerRef3',
 'dl_loadUe_drb3data', 'dl_loadUe_drb3input', 'dl_loadUe_bbBearerRef4',
 'dl_loadUe_drb4data', 'dl_loadUe_drb4input', 'dl_loadUe_bbBearerRef5',
 'dl_loadUe_drb5data', 'dl_loadUe_drb5input', 'dl_loadUe_bbBearerRef6',
 'dl_loadUe_drb6data', 'dl_loadUe_drb6input', 'dl_loadUe_bbBearerRef7',
 'dl_loadUe_drb7data', 'dl_loadUe_drb7input', 'dl_loadUe_wbUsedNbOverridden',
 'dl_loadUe_pucchFormatType', 'dl_loadUe_linkAdaptationUeMode',
 'dl_loadUe_isDrxEnabled', 
 #--------
 'nbm_loadUe_rsrp02', 'nbm_loadUe_rsrp03',
 'nbm_loadUe_rsrp12', 'nbm_loadUe_rsrp13']

drop_before_learning_cols = just_for_reference_cols + const_val_cols + non_gen_cols

rfe_rf_selec_10feats = ['macSduInBytes', 'BSRestimate', 'preamblePwr', 'deltaIcc',
       'iccAchievable', 'srWeight', 'ulRequestTypeBitmap', 'sinrAchievable',
       'NBeamRsrpCurrent', 'tbSizeInBits'] 

# these features are indices, IDs or time indicating metrics and should be removed from the UL sched logs 
#drop_feat_list = ['absTime', 'timeStamp', 'ver', 'esfn', 'slot', 'bbCellIndex', 'cellId', 'beamTableId', 
#                   'bbUeRef', 'WBeamIndexCurrent', 'NBeamIndexCurrent', 'WBeamIndexChosen', 'NBeamIndexChosen', 
#                   'feedbackIndex', 'dlCcIndex', 'antennaPorts', 'pucchSfn', 'WBeamIndexNewBest', 'beamIndex', 
#                   'isPrimaryCell', 'carrierAggregationUsed']

# These are the set of 13 features that were sent to me by Caner. 
ul_lim_feats = ['BSRestimate', 'NBeamRsrpCurrent', 'WBeamRsrpCurrent', 'deltaIcc', 'macSduInBytes', 
                'measNumOfPrb', 'postEqSinr0', 'postEqSinr1', 'powerHeadRoomIndex', 
                'sinrAchievable', 'srWeight', 'tbSizeInBits', 'ulRequestTypeBitmap']

# These features are summed and aggregated over a window and NOT averaged and hence will be padded with zero when there is no observation in a window   
zero_pad_sum_feats = ['macSduInBytes', 'measNumOfPrb', 'tbSizeInBits']
# These features are average over a window and hence can be forward filled into empty windows when there is no observation made.  
ffill_pad_mean_feats = ['BSRestimate', 'NBeamRsrpCurrent', 'WBeamRsrpCurrent', 'deltaIcc', 
                        'postEqSinr0', 'postEqSinr1', 'powerHeadRoomIndex', 
                        'sinrAchievable', 'srWeight', 'ulRequestTypeBitmap']



# This is what I am calling knowledge based selection 
knowledge_based_features = ['ul_chipsetType', 'ul_loadUe_chipsetType', 
             'ul_BSRestimate', 'ul_loadUe_BSRestimate', 
             'ul_numOfPrbs', 'ul_loadUe_numOfPrbs',
            'ul_ACK', 'ul_loadUe_ACK',
             'ul_NBeamRsrpCurrent', 'ul_loadUe_NBeamRsrpCurrent',
             'ul_WBeamRsrpCurrent', 'ul_loadUe_WBeamRsrpCurrent',
             'ul_RI', 'ul_loadUe_RI',
             'ul_ulRequestTypeBitmap', 'ul_loadUe_ulRequestTypeBitmap',
             'ul_ulschIndicator', 'ul_loadUe_ulschIndicator',
             'ul_csiRequest', 'ul_loadUe_csiRequest',
             'ul_macSduInBytes', 'ul_loadUe_macSduInBytes', 
             'ul_measNumOfPrb', 'ul_loadUe_measNumOfPrb',
             'ul_weightBand', 'ul_loadUe_weightBand',
             #'ul_numOfLayers', 'ul_loadUe_numOfLayers',
             #'ul_slot', 'ul_loadUe_slot'
            ] 

# regression
IN_PARAM_R = {
    'model_save_num': 0,
    'classification': False,
    'model_type': 'None',
    'time_wind_size': '10ms',
    'loss': 'mse',
    'eval_metric': 'mse',
    'metrics': ['mse'],
    'out_activation': 'relu', 
    'num_layers': 3,
    'epochs': 300,
    'batch_size': 32,
    'rand_seed': 13,
    'train_size': 0,
    'test_size': 0
}

# classification
IN_PARAM_C = {
    'model_save_num': 0,
    'classification': True,
    'model_type': 'None',
    'time_wind_size': '10ms',
    'loss': 'binary_crossentropy', 
    'eval_metric': 'accuracy',
    'val_metrics': 'val_accuracy',
    'out_activation': 'sigmoid',
    'num_layers': 3,
    'epochs': 600,
    'batch_size': 32,
    'rand_seed': 13,
    'train_size': 0,
    'test_size': 0,
    'class_weights': None,
    'imbalance_handling_method': 0
}

OUT_PARAM_R = {
    'runtime': 0,
    'train_err': 0,
    'test_err': 0
}

# classification
OUT_PARAM_C = {
    'runtime': 0,
    'train_accuracy': 0,
    'train_precision': 0,
    'train_recall': 0,
    'train_f1score': 0,
    'test_accuracy': 0,
    'test_precision': 0,
    'test_recall': 0,
    'test_f1score': 0
}

# taking num of steps as input create step indexed column names for a set of columns
# This is currently used to expand the drop_cols set depending of whether one needs 
# to drop just one or multiple steps of this column 
def expand_cols_to_step_size(cols, steps):
    if steps == 1:
        return cols
    #else
    expanded_cols = []
    for col in cols:
        expanded_cols.extend([col+'_'+str(i) for i in range(steps)])
    #print('debug: expanded cols for the drop cols ')
    #print(expanded_cols)    
    return expanded_cols

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def dist_penalty_obj(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_rows = y_pred.shape[0]
    num_classes = y_pred.shape[1]
    penalty_matrix = np.array([
        [50,7,4,4,1],
        [7,10,7,4,1],
        [4,7,50,7,1],
        [4,4,7,50,1],
        [1,1,1,1,50]])

    #penalty_matrix = np.ones_like(penalty_matrix)
    
    weights = np.ones((num_rows, 1), dtype=float)

    grad = np.zeros((num_rows, num_classes), dtype=float)
    hess = np.zeros((num_rows, num_classes), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(y_pred.shape[0]):
        p = softmax(y_pred[r, :])
        weights[r] = penalty_matrix[int(y_true[r]), int(np.argmax(p))]
        for c in range(y_pred.shape[1]):
            assert y_true[r] >= 0 or y_true[r] <= num_classes
            g = p[c] - 1.0 if c == y_true[r] else p[c]
            g = g * weights[r]
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            grad[r, c] = g
            hess[r, c] = h

    # Right now (XGBoost 1.0.0), reshaping is necessary
    grad = grad.reshape((num_rows * num_classes, 1))
    hess = hess.reshape((num_rows * num_classes, 1))
    return grad, hess



# Custom MAPE loss function
#def mape_objective_function(preds, dtrain):
#    labels = dtrain.get_label()
def mape_obj(labels: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #grad = np.sign(preds - labels) / (labels + 1e-8)
    #hess = 1.0 / (labels + 1e-8)
    grad = (preds - labels) / (0.2 + labels * np.abs(preds - labels))
    hess = 0.1 + np.zeros(len(preds))
    return grad, hess

# Custom evaluation metric (optional)
def mape_eval(preds, dtrain):
    labels = dtrain.get_label()
    error = np.mean(np.abs((labels - preds) / (labels + 1e-8)))
    return 'mape', error

def compute_error(err_type, pred, truth):
    if err_type == 'mae':
        return mean_absolute_error(pred, truth)
    elif err_type == 'mape':
        return mean_absolute_percentage_error(pred, truth)
    else:# assuming it is mse
        return mean_squared_error(pred, truth)

def get_xgb_model(IN_PARAM):
    if IN_PARAM['classification']:
        if IN_PARAM['imbalance_handling_method'] == 'classWeights':
            weight_ratio = IN_PARAM['class_weights'][1] / IN_PARAM['class_weights'][0]
            model = XGBClassifier(n_jobs=30, scale_pos_weight=weight_ratio)
        else:
            if IN_PARAM['loss'] == 'dist_penalty':
                print('using classification loss objective: dist_penalty')
                model = XGBClassifier(n_jobs=30, use_label_encoder=False, objective=dist_penalty_obj)
            elif IN_PARAM['loss'] == 'class_mse':
                # using regression to preserve class order importance 
                # Take round after to attain class 
                model = XGBRegressor(n_jobs=30, seed=IN_PARAM['rand_seed'])
            else: # default
                print('default classification loss objective')
                model = XGBClassifier(n_jobs=30, use_label_encoder=False)
    else:
        print('rand seed used: ', IN_PARAM['rand_seed'])
        if IN_PARAM['loss'] == 'mae':
            #'reg:absoluteerror'
            #params = {'objective':'reg:pseudohubererror'}
            model = XGBRegressor(**params, n_jobs=30, seed=IN_PARAM['rand_seed'])
        elif IN_PARAM['loss'] == 'mape':
            model = XGBRegressor(n_jobs=30, seed=IN_PARAM['rand_seed'], 
                            objective=mape_obj)
                            #eval_metric=mean_absolute_percentage_error,
        else:# use the default which is mse
            model = XGBRegressor(n_jobs=30, seed=IN_PARAM['rand_seed'])
            

    return model

def get_rf_model(IN_PARAM):
    if IN_PARAM['classification']:
        if IN_PARAM['imbalance_handling_method'] == 'classWeights':
            print(IN_PARAM['class_weights'])
            model = RandomForestClassifier(n_jobs=10, class_weight=IN_PARAM['class_weights'])
        else:
            model = RandomForestClassifier(n_jobs=10)
    else:
        model = RandomForestRegressor(n_jobs=10)
        
    return model

def get_mlp_model(n_inputs, n_outputs, IN_PARAM):
    model = Sequential ()
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(15, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(10, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(5, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation = IN_PARAM['out_activation']))
    optimizer = Adam(lr=0.01)
    model.compile(loss=IN_PARAM['loss'], optimizer=optimizer, metrics=IN_PARAM['metrics'])
    
    return model

def get_lstm_model(n_in_feat, n_in_time, n_output, IN_PARAM):
    model = Sequential()
    model.add(LSTM(30, return_sequences=True, input_shape=(n_in_feat, n_in_time)))
    #model.add(LSTM(15, return_sequences=True))
    model.add(LSTM(5))
    model.add(Dense(n_output, activation=IN_PARAM['out_activation']))
    model.compile(loss=IN_PARAM['loss'], optimizer='adam', metrics=IN_PARAM['metrics'])
    
    return model

def normalize (train, test):
    val_scaler = MinMaxScaler()
    val_scaler.fit(train)
    train = val_scaler.transform(train).copy()
    test = val_scaler.transform(test).copy()

    return train, test, val_scaler

def evaluate_model (X_train, X_test, y_train, y_test, model_save_path, IN_PARAM, sample_weights=None):
    if IN_PARAM['model_type'] == 'lstm':
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        model = get_lstm_model(X_train.shape[1], X_train.shape[2], y_train.shape[1], IN_PARAM)
    elif IN_PARAM['model_type'] == 'mlp': 
        model = get_mlp_model(X_train.shape[1], y_train.shape[1], IN_PARAM)
    elif IN_PARAM['model_type'] == 'rf':
        model = get_rf_model(IN_PARAM)
    elif IN_PARAM['model_type'] == 'xgb':
        model = get_xgb_model(IN_PARAM)
    else:    
        print('Do not know model')
        
    if (IN_PARAM['model_type'] == 'rf') or (IN_PARAM['model_type'] == 'xgb'):
        print(sample_weights)
        history = model.fit(X_train,y_train,sample_weight=sample_weights)
    else:
        history = model.fit(X_train,y_train,epochs=IN_PARAM['epochs'], 
                        batch_size=IN_PARAM['batch_size'], validation_data=(X_test,y_test), 
                        shuffle=False, verbose=0, class_weight=IN_PARAM['class_weights'])
                        #class_weight=class_weight)
    #yhat_train = model.predict(X_train)
    #yhat_test = model.predict(X_test) 
    #scores_train = 0
    #scores_test = 0
    #if  IN_PARAM['model_type'] is not 'rf':
    #    scores_train = model.evaluate(X_train, y_train, return_dict=True)
    #    scores_test = model.evaluate(X_test, y_test, return_dict=True)
    
    return model, history    
        
def plot_model_train_info (history, model_save_path, IN_PARAM):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()
    plt.savefig(model_save_path + 'loss_' + IN_PARAM['model_type'] 
                + '_' + str(IN_PARAM['time_wind_size']) 
                + str(IN_PARAM['epochs']) + 'epochs.pdf')
   
    #plt.figure(2)
    #plt.plot(history.history[IN_PARAM['metrics']])
    #plt.plot(history.history[IN_PARAM['val_metrics']])
    #plt.title('model mse')
    #plt.ylabel('mse')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'])
    #plt.show()
    #plt.savefig(model_save_path + 'loss_' + IN_PARAM['model_type'] 
    #            + '_' + str(IN_PARAM['time_wind_size'])
    #            + str(IN_PARAM['epochs']) + 'epochs.pdf')
    return True

def plot_y_yhat (y_train,yhat_train,y_test,yhat_test, model_save_path, IN_PARAM):
    plt.figure(3, figsize=(35, 5))
    plt.plot(yhat_train, label='prediction')
    plt.plot(y_train, label='ground truth')
    plt.plot(y_train-yhat_train, label='error (truth-pred)')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Train samples')
    plt.ylabel('UL delay (ms)')
    plt.xlabel('Samples')
    plt.ylim(-100, 100)
    plt.legend()
    plt.show()
    #plt.savefig(model_save_path + 'train_y_yhat_' + IN_PARAM['model_type'] 
    #            + '_' + str(IN_PARAM['time_wind_size'])
    #            + str(IN_PARAM['epochs']) + 'epochs.pdf')
    
    plt.figure(4, figsize=(35, 5))
    plt.plot(yhat_test, label='prediction')
    plt.plot(y_test, label='ground truth')
    plt.plot(y_test-yhat_test, label='error (truth-pred)')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Test samples')
    plt.ylabel('UL delay (ms)')
    plt.xlabel('Samples')
    plt.ylim(-100, 100)
    plt.legend()
    plt.show()
    #plt.savefig(model_save_path + 'test_y_yhat_' + IN_PARAM['model_type'] 
    #            + '_' + str(IN_PARAM['time_wind_size'])
    #            + str(IN_PARAM['epochs']) + 'epochs.pdf')
    
    return True

def plot_heatmap(windowed_combined_data, plot_name):
    plt.rcParams["figure.autolayout"] = True
    corr_data = windowed_combined_data.corr(method='spearman')
    fig = plt.figure(figsize=(11,11))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    mask = mask[1:, :-1]
    corr = corr_data.iloc[1:,:-1].copy()
    hmap = sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True, mask=mask, 
                       vmax=1, annot_kws={"fontsize":15}, linecolor='black')
    fig = hmap.get_figure()
    plt.yticks(rotation=45)
    fig.autofmt_xdate(rotation=45)
    #plt.savefig(plot_name)
    fig.show()

