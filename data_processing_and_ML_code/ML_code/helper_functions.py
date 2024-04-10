from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from keras.layers import Input, Dense, Lambda
import tensorflow.keras.backend as K

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow.keras.metrics import MeanAbsolutePercentageError


from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer(norm='l2')

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

import pickle

from xgboost import XGBRegressor
from xgboost import XGBClassifier
#from xgboost import DMatri

plt.rcParams.update({'font.size': 22})

# regression
IN_PARAM_R = {
    'classification': False,
    'model_type': 'None',
    'time_wind_size': '10ms',
    'loss': 'mse',
    'eval_metric': 'mape',
    'metrics': ['mape'], 
    'out_activation': 'linear', 
    'num_layers': 3,
    'epochs': 500,
    'batch_size': 64,
    'learning_rate': 0.0001,
    'rand_seed': 13,
    'train_size': 0,
    'test_size': 0
}

# classification
IN_PARAM_C = {
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
    'train_err_mape': 0,
    'test_err_mape': 0
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

#=============================================
# Data Manipulation Functions
#=============================================

# Takes a pandas df as input and selects samples that belong to a slice
def data_slice_filter_samples(slice, df, network_info):
    if slice == 'macro':
        # separate all the samples where cellId is in the macro_cells list
        filtered_df = df[df['cellId'].isin(network_info['macro_cells'])]
    elif slice == 'micro':
        # separate all the samples where cellId is in the micro_cells list
        filtered_df = df[df['cellId'].isin(network_info['micro_cells'])]
    elif slice == 'fast':
        # get all the fast IMSIs
        filtered_df = df[df['IMSI'].isin(network_info['fast_imsis'])]
    elif slice == 'slow':
        filtered_df = df[df['IMSI'].isin(network_info['slow_imsis'])]
    elif slice == 'all':
        # keep all return the whole slice
        filtered_df = df
    else:
        print('UNKNOWN SLICE NAME')
        filtered_df = df
    #print(filtered_df[['IMSI', 'cellId']].head(n=10) )
    return filtered_df

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



#=============================================
# Loss Functions and Error Functions
#=============================================

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

# Take as input, class labels and then convert them to delay values by mapping them to 
# the mid-points of the bins. This way I can compute mae from class labels   
def class_mae(yhat, y, num_classes, delay_class_edges):
    # find the mid points of bins from the bin edges
    class_vals = (delay_class_edges + np.roll(delay_class_edges, 1))/2
    class_vals = class_vals[1:num_classes+1]
    # replace bin class labels with corresponding bin midpoints  
    yhat_val = [class_vals[int(i)] for i in yhat]
    y_val = [class_vals[int(i)] for i in y]
    # compute mae with these mid points
    return mean_absolute_error(yhat_val, y_val)

# Take as input class labels in continuous form and convert them to class labels 
# Then compute the confusion matrix 
def value_to_class_label(vals, delay_class_edges):
    delay_class_indices = np.digitize(vals, delay_class_edges)
    return (delay_class_indices - 1)

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

def compute_error(truth, pred, err_type):
    if err_type == 'mae':
        return mean_absolute_error(truth, pred)
    elif err_type == 'mape':
        return mean_absolute_percentage_error(truth, pred)
    else:# assuming it is mse
        return mean_squared_error(truth, pred)


#========================================
# Pretraining functions
#========================================

# Takes a pandas df as input and sets up dae pretraining. Returns the model  
def pretrain_with_dae(df, hypp_ssl_dae):
    # initialize a dae model   
    if hypp_ssl_dae['arch'] == 'deepstack':
        dae = DAE(
            body_network=hypp_ssl_dae['arch'],
            body_network_cfg=dict(hidden_size=hypp_ssl_dae['hid_size'], num_layers=hypp_ssl_dae['num_layers']),
            swap_noise_probas=hypp_ssl_dae['swap_noise_prob'],
            device='cuda')
    elif hypp_ssl_dae['arch'] == 'deepbottleneck':
        dae = DAE(
            body_network=hypp_ssl_dae['arch'],
            body_network_cfg=dict(hidden_size=hypp_ssl_dae['hid_size'], num_layers=hypp_ssl_dae['num_layers'], bottleneck_size=hypp_ssl_dae['bottle_neck_size']),
            swap_noise_probas=hypp_ssl_dae['swap_noise_prob'],
            device='cuda')
    # fit the model
    dae.fit(df, verbose=1, optimizer_params={'lr': 3e-4}, batch_size=hypp_ssl_dae['batch_size'])
    return dae

# Takes a pandas df as input and sets up tabnet pretraining. Returns the model  
def pretrain_with_tabnet(X_ssl_train, X_ssl_val, hypp_ssl_tabnet):
    # initialize a tabnet model   
    # TabNetPretrainer
    ssl_model = TabNetPretrainer(
        optimizer_fn=Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type=hypp_ssl_tabnet['mask_type'],
        n_a = hypp_ssl_tabnet['n_da'], 
        n_d = hypp_ssl_tabnet['n_da'], 
        n_steps = hypp_ssl_tabnet['n_steps'],
        n_independent=hypp_ssl_tabnet['n_independent'],
        n_shared=hypp_ssl_tabnet['n_shared'],
        n_shared_decoder=hypp_ssl_tabnet['n_shared_decoder'],
        n_indep_decoder=hypp_ssl_tabnet['n_indep_decoder']
    )
    ssl_model.fit(
        X_train=X_ssl_train.values,
        eval_set=[X_ssl_val.values],
        pretraining_ratio=hypp_ssl_tabnet['noise_ratio'],
        batch_size=hypp_ssl_tabnet['batch_size'],
        max_epochs=hypp_ssl_tabnet['max_epochs'],
        patience=hypp_ssl_tabnet['patience']
    )
    return ssl_model


#==================================================
# Scalers, Models and Training functions 
#==================================================

# We need to try different Scalers
def create_scaler (train, scaler_type):
    if scaler_type == 'minmax':
        val_scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        val_scaler = StandardScaler()
    elif scaler_type == 'robust':
        val_scaler = RobustScaler()
    elif scaler_type == 'maxabs':
        val_scaler = MaxAbsScaler()
    elif scaler_type == 'l2norm':
        val_scaler = Normalizer(norm='l2')
    else:
        print('Unknown scaler type')
        
    val_scaler.fit(train)
    return val_scaler

def get_xgb_model(IN_PARAM, X_train, y_train, X_val, y_val, model_to_save_name, hyper_params):
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

    # fit the model
    history = model.fit(X_train, y_train)

    # save the model
    model.save_model(model_to_save_name+'.json')
    
    return model, history

def get_mlp_model(n_inputs, n_outputs, IN_PARAM, X_train, y_train, X_val, y_val, model_to_save_name, hyper_params):
    model = Sequential ()
    for l in range(0,len(hyper_params['fc_layers'])):
        if l==0:
            model.add(Dense(hyper_params['fc_layers'][l], input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        else:
            model.add(Dense(hyper_params['fc_layers'][l], kernel_initializer='he_uniform', activation='relu'))
    
    model.add(Dense(n_outputs, activation = IN_PARAM['out_activation']))
    
    optimizer = Adam(learning_rate=IN_PARAM['learning_rate'])
    if IN_PARAM['metrics'][0] == 'mape':
        model.compile(loss=IN_PARAM['loss'], 
                      optimizer=optimizer, 
                      metrics=[MeanAbsolutePercentageError()])
    else:
        model.compile(loss=IN_PARAM['loss'], optimizer=optimizer, metrics=IN_PARAM['metrics'])

    # Fit the model 
    early_stopping = EarlyStopping(monitor='val_loss', patience=hyper_params['patience'], restore_best_weights=True)
    history = model.fit(X_train,y_train, 
                        epochs=hyper_params['epochs'], 
                        batch_size=hyper_params['batch_size'], 
                        validation_data=(X_test,y_test), 
                        callbacks=[early_stopping], 
                        shuffle=True, 
                        verbose=1)

    model.save(model_to_save_name)
    
    return model, history

def get_tabnet_model(IN_PARAM, X_train, y_train, X_val, y_val, model_to_save_name, hyper_params):
    model = TabNetRegressor()
    history = model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        batch_size=hyper_params['batch_size'],
        max_epochs=hyper_params['max_epochs'],
        patience=hyper_params['patience']
    )

    # save the model 
    model.save_model(model_to_save_name)
    
    return model, history

def train_model (X_train, X_val, y_train, y_val, IN_PARAM, model_to_save_name, hyper_params, sample_weights=None, save_str=[]):
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    print('model type', IN_PARAM['model_type'])
    if IN_PARAM['model_type'] == 'mlp': 
        model, history = get_mlp_model(X_train.shape[1], y_train.shape[1], IN_PARAM, X_train, y_train, X_val, y_val, model_to_save_name, hyper_params)
    elif IN_PARAM['model_type'] == 'xgb':
        model, history = get_xgb_model(IN_PARAM, X_train, y_train, X_val, y_val, model_to_save_name, hyper_params)
    elif IN_PARAM['model_type'] == 'tabnet':
        model, history = get_tabnet_model(IN_PARAM, X_train, y_train, X_val, y_val, model_to_save_name, hyper_params)
    else:    
        print('Do not know model')

    return model, history    


#==========================================
# Plotting functions
#==========================================

def plot_model_train_info (history, model_save_path, IN_PARAM):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
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

def tabnet_explain(model, X, feat_filter, X_feats):
    # Global explainability : feat importance summing to 1
    print('Global explainability')
    importances = model.feature_importances_
    plot_feature_importance(importances, X_feats, feat_filter)
    
    # Local explainability and masks
    print('Local explainability')
    explain_matrix, masks = model.explain(X)
    print('Explainability Matrix')
    print(explain_matrix)
    fig, axs = plt.subplots(1, 3, figsize=(20,20))
    for i in range(3):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
    return

def xgb_explain(model, feat_filter, X_feats):
    # The length of importances reflects the number of features used
    importances = model.feature_importances_
    plot_feature_importance(importances, X_feats, feat_filter)
    return

def plot_feature_importance(importances, X_feats, feat_filter):
    # increasing order in value and hence decreasing order in importance 
    # sort the importances and then fetch the index value of those importances 
    indices = np.argsort(importances)
    #This is in ascending order of 
    bar_vals = importances[np.flip(indices)[0:feat_filter]]
    bar_names = X_feats[np.flip(indices)[0:feat_filter]]
    #print(importances[np.flip(indices)[0:feat_filter]])
    print(X_feats[np.flip(indices)[0:feat_filter]])

    #top_n_features = list( set(top_n_features).union(set(bar_names)))
    #print('Top n feature list: ', top_n_features)
    plt.figure()
    plt.barh(range(len(bar_vals)), np.flip(bar_vals), color='b', align='center')
    plt.yticks(range(len(bar_vals)), np.flip(bar_names))

    plt.title('Feature importance')
    plt.xlabel('Relative Importance')
    #plt.savefig('plots_for_paper/feat_imp'+str(IN_PARAM['rand_seed'])+'.pdf', bbox_inches='tight')
    plt.show() 
    
    return
#=============================================
# DANN 
#=============================================

# Define the domain classifier with a gradient reversal layer
def gradient_reversal(x):
    return -x

# Define loss functions
def custom_regression_loss(y_true, y_pred):
    # Define your regression loss function here (e.g., mean squared error)
    return K.mean(K.square(y_true - y_pred))

def domain_adversarial_loss(y_true, y_pred):
    # Binary cross-entropy loss for domain classification
    return K.mean(K.binary_crossentropy(y_true, y_pred))


def dann(input_dim, X_train, y_train_regression, domain_labels):
    # Define the input shape
    input_shape = (input_dim,)

    # Define the feature extractor shared by the regressor and domain classifier
    input_layer = Input(shape=input_shape)
    shared_feature_extractor = Dense(64, activation='relu')(input_layer)  # Adjust the architecture as needed

    # Define the regressor for the regression task
    regressor = Dense(1, activation='linear')(shared_feature_extractor)  # Linear activation for regression

    domain_classifier = Dense(1, activation='sigmoid')(Lambda(gradient_reversal)(shared_feature_extractor))

    # Create the DANN model
    dann_model = Model(inputs=input_layer, outputs=[regressor, domain_classifier])

    # Compile the model
    dann_model.compile(
        optimizer=Adam(learning_rate=IN_PARAM['learning_rate']),
        loss={'regressor': custom_regression_loss, 'domain_classifier': domain_adversarial_loss},
        loss_weights={'regressor': 1.0, 'domain_classifier': 1.0}  # Adjust the weights as needed
    )

    # Train the model
    dann_model.fit(
        x_train,
        {'regressor': y_train_regression, 'domain_classifier': domain_labels},  # domain_labels are 0 for source, 1 for target
        epochs=IN_PARAM['epochs'],
        batch_size=IN_PARAM['batch_size']
    )

    # Make predictions using the regressor
    predictions = dann_model.predict(x_test)[0]  # Index 0 corresponds to the regressor output
