
# NOTE: I am using keras as well as tensorflow.keras which sounds like it would cause problems 
# But I am too afraid to touch this because I struggled with some version problems here before and I want to put this off to later 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import Adam as Adam_tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.callbacks import LambdaCallback

import tensorflow as tf
import tensorflow.keras.backend as K

from keras.layers import Input, Dense, Lambda, BatchNormalization, Dropout, ReLU

#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from typing import Tuple
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import numpy as np
from decimal import *
import json
from csv import DictWriter
import sys
import copy 
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
import random
import pickle
import csv

from xgboost import XGBRegressor
from xgboost import XGBClassifier

import torch
from torch.optim import Adam as Adam_tor
from torch import nn
from torchinfo import summary

from tabular_dae.model import DAE
from tabular_dae.model import load as dae_load

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

#from VIME.vime_self import vime_self
#from VIME.vime_semi_mod import vime_semi as vime_semi_mod
#from VIME.vime_utils import perf_metric


import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchmetrics.regression import R2Score

# TS3L library
from ts3l.utils import TS3LDataModule

# DAE
from ts3l.pl_modules import DAELightning
from ts3l.utils.dae_utils import DAEDataset, DAECollateFN
from ts3l.utils.dae_utils import DAEConfig
# VIME
from ts3l.pl_modules import VIMELightning
from ts3l.utils.vime_utils import VIMEDataset
from ts3l.utils.vime_utils import VIMEConfig
from ts3l.utils.vime_utils import VIMESecondPhaseCollateFN
# SCARF
from ts3l.pl_modules import SCARFLightning
from ts3l.utils.scarf_utils import SCARFDataset
from ts3l.utils.scarf_utils import SCARFConfig
#SUBTAB
from ts3l.pl_modules import SubTabLightning
from ts3l.utils.subtab_utils import SubTabDataset, SubTabCollateFN
from ts3l.utils.subtab_utils import SubTabConfig
from ts3l.functional.subtab import arrange_tensors 
#SWITCHTAB
from ts3l.pl_modules import SwitchTabLightning
from ts3l.utils.switchtab_utils import SwitchTabDataset, SwitchTabFirstPhaseCollateFN
from ts3l.utils.switchtab_utils import SwitchTabConfig

from hyperparameters import *

plt.rcParams.update({'font.size': 22})

M = (10**6)
K = (10**3)


plt.rcParams.update({'font.size': 22})

COLOUR_HEX = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']


# Column names in the dataset that contain the ground truth labels for our prediction tasks of interest
all_learning_tasks_in_data = ['dashClient_trace.txt_newBitRate_bps', # predictable with 0.3516 mape # most imp feature DlPdcpStats.txt_PduSize
                              'dashClient_trace.txt_oldBitRate_bps', 
                              'delay_trace.txt_ul_delay', # predictable with 0.06 mape # most imp feature UlPdcpStats.txt_delay
                              'delay_trace.txt_dl_delay', # predictable with 0.07 mape # most imp feature DlPdcpStats.txt_delay
                              'vrFragment_trace.txt_vr_frag_time', # predictable with 0.0394 mape # most imp feature DlPdcpStats.txt_delay
                              'vrFragment_trace.txt_vr_frag_thput_mbps', # predictable with 0.1150 mape # most imp feature DlPdcpStats.txt_delay
                              'vrFragment_trace.txt_vr_burst_time', # predictable with 0.0869 mape # most imp feature DlPdcpStats.txt_max
                              'vrFragment_trace.txt_vr_burst_thput_mbps', # predictable with 0.2122 mape # most imp feature DlPdcpStats.txt_delay 
                              'httpClientRtt_trace.txt_page_load_time', 
                              'httpClientRtt_trace.txt_webpage_size']

# Drop columns that we should not be keeping from the input features part of the dataset 
sum_cols_substr = ['sizeTb1', 'nTxPDUs', 'TxBytes', 'nRxPDUs', 'RxBytes', 'size']
drop_col_substr = ['sizeTb2', 'mcsTb2', 'layer', 'txMode', 'timeslot', 'frame', 'min', 'max', 'stdDev', '.1', 'LCID', 'rv']

categorical_cols = ['cell_conn_type']
# Need to drop IMSI separately once I am done using it 

# Read these from the ue_groups file later and also rename ue_groups to something else
network_info={}
network_info['total_num_ues'] = 90
network_info['macro_cells'] = [1,2,3]
network_info['micro_cells'] = [4,5,6]
network_info['macro_imsis'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
network_info['fast_imsis'] = [1,2,3,4,5,6]
network_info['only_delay_imsis'] = [3,5,7,9,13,15,17,19,23,25,27,29,33,35,37,39,43,45,47,49,53,55,57,59,63,65,67,69,73,75,77,79,83,85,87]
network_info['micro_imsis'] = list(set(range(1, network_info['total_num_ues']+1)) - set(network_info['macro_imsis']))

bitrate_levels = [45000, 89000, 131000, 178000, 221000, 263000, 334000, 396000, 522000, 595000, 791000, 1033000, 1245000, 
                  1547000, 2134000, 2484000, 
                  3079000, 3527000, 3840000]



# A weighted loss class for pytorch to use 
class WeightedL1Loss(nn.Module):
    def __init__(self, bin_edges, bin_weights):
        """
        :param bin_edges: List of bin edges. Must be of length n_bins + 1.
        :param bin_weights: List of weights for each bin. Must be of length n_bins.
        """
        super(WeightedL1Loss, self).__init__()
        assert len(bin_edges) - 1 == len(bin_weights), "Number of bins must match number of weights"
        self.bin_edges = torch.tensor(bin_edges)  # Edges of bins
        self.bin_weights = torch.tensor(bin_weights)  # Weights for each bin

    def forward(self, predictions, targets):
        # Compute the error
        mae_loss = torch.abs(predictions - targets)

        # Find the bin index for each target
        bin_indices = torch.bucketize(targets, self.bin_edges, right=False) - 1  # Bucketize the targets into bins
        
        # Get the weights based on the bin index for each target
        sample_weights = self.bin_weights[bin_indices]
        
        # Apply the weights to the squared errors
        weighted_loss = mae_loss * sample_weights
        
        # Return the mean weighted loss
        return torch.mean(weighted_loss)


# A mape loss class for pytorch to use 
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-8  # to avoid division by zero
        loss = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon)) * 100)
        return loss

class MLPRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, output_dim, use_batchnorm=True, 
                 use_dropout=True, dropout_rate=0.1, learning_rate=0.0001, loss_fn_str='MSELoss', bin_edges=None, bin_weights=None):
        super(MLPRegressor, self).__init__()
        self.save_hyperparameters()  # Saves all the arguments passed to __init__
        
        self.learning_rate = learning_rate
        # Model layers
        layers = []
        # Input to hidden layer
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            # Optionally add BatchNorm layer
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            # ReLU activation
            layers.append(nn.ReLU())
            # Optionally add Dropout layer
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        # Define the model as a sequential container
        self.model = nn.Sequential(*layers)

        print('loss_fn_str', loss_fn_str)
        # loss function to use
        if loss_fn_str == 'MAPELoss':
            self.loss_fn = MAPELoss()
        elif loss_fn_str == 'WeightedL1Loss':
            self.loss_fn = WeightedL1Loss(bin_edges, bin_weights)
        else:
            self.loss_fn = getattr(nn, loss_fn_str)()
        print('USING LOSS FN: ', self.loss_fn)
        
        # Initialize lists to store losses
        self.train_losses = []
        self.val_losses = []
        self.val_mapes = []

        # Initialize variables to accumulate losses within an epoch
        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.val_mape_epoch = []

    
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #loss = F.mse_loss(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.train_loss_epoch.append(loss.item())
        self.log('train_losses', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #val_loss = F.mse_loss(y_hat, y)
        val_loss = self.loss_fn(y_hat, y)
        val_mape = torch.mean(torch.abs((y_hat - y) / y)) * 100
        self.val_loss_epoch.append(val_loss.item())
        self.val_mape_epoch.append(val_mape.item())
        self.log('val_losses', val_loss, on_step=False, on_epoch=True)
        self.log('val_mapes', val_mape, on_step=False, on_epoch=True)
        return val_loss
        
    def on_train_epoch_end(self):
        # Calculate and store the average training loss for the epoch
        avg_train_loss = np.mean(self.train_loss_epoch)
        self.train_losses.append(avg_train_loss)
        self.train_loss_epoch = []  # Reset for the next epoch

    def on_validation_epoch_end(self):
        # Calculate and store the average validation loss for the epoch
        avg_val_loss = np.mean(self.val_loss_epoch)
        self.val_losses.append(avg_val_loss)
        self.val_loss_epoch = []  # Reset for the next epoch
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class MyProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=10, leave = True):
        super().__init__()
        
        self._refresh_rate = refresh_rate
        self._leave = leave
        print('refresh rate: ', refresh_rate)
        
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar
        

def initialize(r=None):
    if r is None:
        r = np.random.randint(0, 1000)
    print('Run with random seed: ', r,)    
    random.seed(r)
    np.random.seed(r)
    tf.random.set_seed(r)
    torch.manual_seed(r)
    torch.cuda.manual_seed_all(r)
    
    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("No GPU found.")
    print(torch.cuda.device_count())

# Function to format arrays in DataFrame cells to 3 significant digits
def format_array(arr):
    return np.array2string(np.array(arr), formatter={'float_kind':lambda x: "%.3f" % x})

def mean_array(arr):
    return np.mean(arr)

def std_array(arr):
    return np.std(arr)
    
# Function to create an array of size EXP_PARAM['num_rand_runs'] filled with NaNs
def create_nan_array(size):
    return[float('nan')] * size
    #return np.full(size, np.nan)

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
    return filtered_df

def add_cell_type_as_feature(data, network_info):
    data['cell_conn_type'] = 0 # 0 is micro cell 
    data.loc[data['cellId'].isin(network_info['macro_cells']), 'cell_conn_type'] = 1 # 1 is macro cell
    data = data.drop(['cellId'], axis=1, errors='ignore')
    return data

def impute_data(data, x_cols, method, sum_cols_substr): 
    x_data = data[x_cols]
    if method == 'forward_fill':
        cols_to_not_impute = [data_col for data_col in x_data.columns if any(sum_substr in data_col for sum_substr in sum_cols_substr)]
        cols_to_impute = list(set(x_data.columns) - set(cols_to_not_impute))
        x_data[cols_to_not_impute] = x_data[cols_to_not_impute].fillna(0)
        x_data[cols_to_impute] = x_data[cols_to_impute].fillna(method='ffill')
    elif method == 'zero_fill':
        x_data = x_data.fillna(0)
    else:
        print('Do not know how to impute with ', method, ' method')
    
    data[x_cols] = x_data
    
    return data

# Shift learning_task columns to the nextWindow for nextWindow prediction
def nextWindow_shift(df, tasks):
    df[tasks] = df[tasks].shift(periods=1)
    return df 


# Read the dataset
def read_and_concatenate_runs(concat_runs, dataset_folder, slice, network_info, time_step_size, use_all_feats, drop_col_substr, 
                              learning_tasks, shift_samp_for_predict, impute_method, sum_cols_substr, all_learning_tasks_in_data):
    
    data = pd.DataFrame()
    print('Concatenating runs: ', concat_runs)
    for run in concat_runs:
        csv_read_start_time = time.time()
        this_run_data = pd.read_csv(dataset_folder+'run'+str(run)+'_dataslice_all_webpage_video_delay_vr_'+time_step_size+'.csv', delimiter=",")
        print('Time to read csv file for run: ', time.time() - csv_read_start_time)
        # separate the rows that belong to the desired data slice
        this_run_data = data_slice_filter_samples(slice, this_run_data, network_info)
        # Drop columns that we do not want to include in the dataset
        if use_all_feats:
            # Dropping unwanted columns 
            cols_to_drop = [data_col for data_col in this_run_data.columns if any(drop_substr in data_col for drop_substr in drop_col_substr)]
            this_run_data = this_run_data.drop(cols_to_drop, axis=1, errors='ignore')
        else:
            this_run_data = this_run_data[top_n_agg]

        # Dropping columns that are only nans
        this_run_data = this_run_data.drop(this_run_data.columns[this_run_data.isna().all(axis=0)], axis=1, errors='ignore')
        # Impute the X part of the data
        x_cols = list(set(this_run_data.columns) - set(all_learning_tasks_in_data))
        this_run_data_imputed = this_run_data.groupby(by=['IMSI']).apply(lambda group: impute_data(group, x_cols, impute_method, sum_cols_substr)).droplevel(level='IMSI') 
        # Shift the learning_task parts 
        if shift_samp_for_predict:
            this_run_data_imputed = this_run_data_imputed.groupby(by=['IMSI']).apply(lambda group: nextWindow_shift(group, all_learning_tasks_in_data)).droplevel(level='IMSI')
            
        # concatenate over runs
        data = pd.concat([data, this_run_data_imputed], axis=0) 
        print('Loaded run', str(run))

    # drop IMSI and wind_tstamp column since I don't need it to groupby anymore
    data = data.drop(['IMSI', 'wind_tstamp'],axis=1)
    data = add_cell_type_as_feature(data, network_info)
    # indices dont matter once I concatenate data from multiple runs 
    data = data.reset_index(drop=True)

    return data

# expects input as a numpy array 
def bin_and_remove_outliers(data, y, bins, perc, sample_ratio_limit):
    # Remove the samples in both y and data that are above the 99th percentile in y 
    filter_y = y[y <= np.quantile(y, perc)]
    filter_data = data[y <= np.quantile(y, perc)]
    
    strat_arr = np.array(pd.cut(filter_y, bins=bins, labels=False))
    bin_values, bin_counts = np.unique(strat_arr, return_counts=True)
    print('bin_and_remove_outliers')
    print('Before removing bins that are not populated enough')
    print(bin_values, bin_counts)

    bins_to_filter = bin_values[bin_counts <= sample_ratio_limit*len(y)]
    # find the index of samples that are part of this bin with only one sample
    filter_indices = ~np.isin(strat_arr, bins_to_filter)
    filter_strat = strat_arr[filter_indices]
    # Filter the original data accordingly
    filter_data = filter_data[filter_indices]
    filter_y = filter_y[filter_indices]
    print('After removing bins that are not populated enough')
    print(bin_values, bin_counts)
    
    return filter_data, filter_y, filter_strat

def bin_and_filter(data, y, bins, sample_ratio_limit):
    strat_arr = np.array(pd.cut(y, bins=bins, labels=False))
    #print(strat_arr.shape)
    # Count the number of samples in each bin
    bin_values, bin_counts = np.unique(strat_arr, return_counts=True)
    print('bin_and_filter: Before')
    print(bin_values, bin_counts)
    print(np.quantile(y, [0.95,0.97,0.99]))
    # Identify the bins with only one sample
    bins_to_filter = bin_values[bin_counts <= sample_ratio_limit*len(y)]
    # find the index of samples that are part of this bin with only one sample
    filter_indices = ~np.isin(strat_arr, bins_to_filter)
    strat_filtered = strat_arr[filter_indices]
    # Filter the original data accordingly
    filter_data = data[filter_indices]
    filter_y = y[filter_indices]
    # Re-bin the remaining data
    new_strat_arr = np.array(pd.cut(filter_y, bins=bins, labels=False))
    # Count the number of samples in each bin
    bin_values, bin_counts = np.unique(new_strat_arr, return_counts=True)
    print('bin_and_filter: After')
    print(bin_values, bin_counts)
    print(np.quantile(filter_y, [0.95,0.97,0.99]))
    
    return filter_data, filter_y, new_strat_arr

def filter_edge(data, y, sample_ratio_limit):
    # Count the number of samples in each bin
    bin_values, bin_counts = np.unique(y, return_counts=True)
    print('filter_edge: Before')
    print(bin_values, bin_counts)
    # Identify the bins with only one sample
    bins_to_filter = bin_values[bin_counts <= sample_ratio_limit*len(y)]
    # find the index of samples that are part of this bin with only one sample
    filter_indices = ~np.isin(y, bins_to_filter)
    # Filter the original data accordingly
    filter_data = data[filter_indices]
    filter_y = y[filter_indices]
    # Count the number of samples in each bin
    bin_values, bin_counts = np.unique(filter_y, return_counts=True)
    print('filter_edge: After')
    print(bin_values, bin_counts)
    
    return filter_data, filter_y

def dash_create_classes(arr, bitrate_steps):
    arr = np.array(arr)
    bitrate_steps = np.array(bitrate_steps)
    bitrate_ind = np.arange(0, len(bitrate_steps))
    # Calculate the absolute differences between each element in arr and each element in bin_vals
    differences = np.abs(arr[:, np.newaxis] - bitrate_steps)
    # Find the index of the smallest difference for each element in arr
    nearest_indices = np.argmin(differences, axis=1)
    # Use these indices to select the corresponding values in bin_vals
    rounded_arr = bitrate_steps[nearest_indices]
    rounded_arr_bin_ind = bitrate_ind[nearest_indices]
    print(np.unique(rounded_arr, return_counts=True))
    new_rounded_arr_bin_ind = [0 if val in np.arange(0, 13) else val for val in rounded_arr_bin_ind] 
    new_rounded_arr_bin_ind = [1 if val in np.arange(13, 16) else val for val in new_rounded_arr_bin_ind]
    new_rounded_arr_bin_ind = [2 if val in np.arange(16, 19) else val for val in new_rounded_arr_bin_ind]
    print(np.unique(new_rounded_arr_bin_ind, return_counts=True))
    
    return new_rounded_arr_bin_ind

def get_cont_and_cat_cols(df):
    continuous_cols = []
    categorical_cols = []
    for col in df.columns:
        if df[col].nunique() < 20:  # Arbitrary threshold for discrete numeric columns
            categorical_cols.append(col)
            print(col)
            print(df[col].unique())
        else:
            continuous_cols.append(col)

    print('categorical_cols: ', categorical_cols)

    return continuous_cols, categorical_cols


def make_data_pretrain_ready (pretrain_data: pd.DataFrame, traffic_cols):
    # Drop the rows that do not represent an instance where there is traffic in the network 
    # from the perspective of the traffic_cols columns  
    print('pretrain_data, before removing rows that dont have traffic ', pretrain_data.shape)
    pretrain_data = pretrain_data.dropna(subset=traffic_cols, how='all', axis=0)
    print('pretrain_data, after removing rows that dont have traffic ', pretrain_data.shape)
    
    return pretrain_data
    

def make_data_sup_model_ready(data, learning_task, learning_task_type, all_learning_tasks_in_data, bitrate_levels, clip_outliers, delay_clip_th):
    # Prepare the train and test sets and drop rows which dont have ground truth 
    print('Dropping rows for which this learning_tasks label is NA, since there is no ground truth')
    data_na_dropped = data.dropna(subset=[learning_task])
    print(data_na_dropped.shape)

    # Create classes from the bitrate levels 
    if learning_task == 'dashClient_trace.txt_newBitRate_bps' or learning_task == 'dashClient_trace.txt_oldBitRate_bps':
        data_na_dropped[learning_task] = dash_create_classes(data_na_dropped[learning_task], bitrate_levels)

    # Separate the X and the y from the data 
    y = data_na_dropped[learning_task]
    X = data_na_dropped.drop(all_learning_tasks_in_data, axis=1, errors='ignore')

    # If the beginning of the dataset has nans and we are using ffill then there might still be nans in the data 
    # so drop any additional rows with nans that have remained because of this
    X = X.dropna()
    # Do the same for y as well                 
    y = y.loc[X.index]

    if (learning_task in ['delay_trace.txt_ul_delay', 'delay_trace.txt_dl_delay']) and clip_outliers:
        print('NOTE: clipping all rows with delay values > ', delay_clip_th)
        y.loc[y > delay_clip_th] = delay_clip_th
        #print('NOTE: dropping all rows with delay values > ', delay_drop_th)
        #data = data[data['owd_ul'] <= delay_drop_th]
    
    # Save the columns to use for feature importance graphs
    X_feats = X.columns.to_numpy()
    
    #continuous_cols, categorical_cols = get_cont_and_cat_cols(X)
    
    #np.savetxt('input_features_list.csv', X_feats, delimiter=',', fmt="%s")
    #np.savetxt('continuous_input_features.csv', continuous_cols, delimiter=',', fmt="%s")
    #np.savetxt('categorical_input_features.csv', categorical_cols, delimiter=',', fmt="%s")

    # Convert everything to numpy because I am assuming this in bin_and_remove_outliers
    # Would be better to keep it as pandas to avoid unnecessary changes to data format, but I would need to change the code 
    # so... it stays like this for now 
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    
    # Sample a certain number of labeled samples from the train set to use in fine-tuning
    # This step needs to be done after all operations that could drop rows are done 
    if learning_task_type == 'reg':
        print('NOTE: removing samples that are above the 99th percentile')
        print('NOTE: stratifying regression samples using 5 bins')
        X_np, y_np, strat_array = bin_and_remove_outliers(X_np, y_np, bins=5, perc=0.99, sample_ratio_limit=0.01)
    else: # 'clas'
        strat_array = y_np

    X_pd = pd.DataFrame(X_np, columns=X_feats)
    
    return X_pd, y_np, strat_array, X_feats
            
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
    if err_type == 'mse':
        return mean_squared_error(truth, pred)
    elif err_type == 'mae':
        return mean_absolute_error(truth, pred)
    elif err_type == 'mape':
        return mean_absolute_percentage_error(truth, pred)
    elif err_type == 'r2':
        return r2_score(truth, pred)
    elif err_type == 'acc':
        return accuracy_score(truth, pred)
    elif err_type == 'f1score':
        return f1_score(truth, pred, average='weighted')# or try 'macro'
    elif err_type == 'roc_auc':
        return roc_auc_score(truth, pred, multi_class='ovo', average='weighted')# or try 'macro'
    else:
        print('ERROR: UNKNOWN ERROR METRIC ', err_type)





#========================================
# TS3L random initialize function
#========================================
def initialize_s3l_model_random(type, input_dim, categorical_cols, continuous_cols):
    
    assert (type in ['s3l_dae', 's3l_vime', 's3l_scarf', 's3l_subtab', 's3l_switchtab'], 
                        f"Invalid pretrain_model_to_load_type: {type}.")
    if type == 's3l_dae':
        hypp = s3l_hyp_ssl_dae
        config = DAEConfig (input_dim=input_dim,
                       hidden_dim=hypp['hidden_dim'], encoder_depth=hypp['encoder_depth'], 
                       noise_type = hypp['noise_type'], noise_ratio = hypp['noise_ratio'], 
                       num_categoricals=len(categorical_cols), num_continuous=len(continuous_cols),
                       # These are specific to tasks and not to pretraining
                       task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                       head_depth = hypp['head_depth'], output_dim=1)
        model = DAELightning(config)
    elif type == 's3l_vime':
        hypp = s3l_hyp_ssl_vime
        config = VIMEConfig(task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                            input_dim=input_dim, hidden_dim=hypp['hidden_dim'], output_dim=1, 
                            encoder_depth = hypp['encoder_depth'], dropout_rate = hypp['dropout_rate'],
                            alpha1=hypp['alpha1'], alpha2=hypp['alpha2'], 
                            beta=hypp['beta'], K=hypp['K'], p_m = hypp['p_m'],
                            num_categoricals=len(categorical_cols), num_continuous=len(continuous_cols))
        model = VIMELightning(config)
    elif type == 's3l_scarf':
        hypp = s3l_hyp_ssl_scarf
        config = SCARFConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={},
                             input_dim=input_dim, hidden_dim=hypp['hidden_dim'],
                             output_dim=1, encoder_depth=hypp['encoder_depth'], head_depth=hypp['head_depth'],
                             dropout_rate=hypp['dropout_rate'], corruption_rate = hypp['corruption_rate']) 
        model = SCARFLightning(config)
    elif type == 's3l_subtab':
        hypp = s3l_hyp_ssl_subtab
        config = SubTabConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                              input_dim=input_dim, hidden_dim=hypp['hidden_dim'], 
                              output_dim=1, tau=hypp['tau'], use_cosine_similarity=hypp['use_cosine_similarity'], use_contrastive=hypp['use_contrastive'], 
                              use_distance=hypp['use_distance'], n_subsets=hypp['n_subsets'], overlap_ratio=hypp['overlap_ratio'],
                              mask_ratio=hypp['mask_ratio'], noise_type=hypp['noise_type'])
        model = SubTabLightning(config)
    elif type == 's3l_switchtab':
        hypp = s3l_hyp_ssl_switchtab
        config = SwitchTabConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                                     input_dim=input_dim, hidden_dim=hypp['hidden_dim'], output_dim=1, 
                                     encoder_depth=hypp['encoder_depth'], n_head = hypp['n_head'], u_label = hypp['u_label'])
        model = SwitchTabLightning(config)
    return model


#========================================
# Pretraining functions
#========================================
def get_checkpoint (save_path):
    checkpoint_callback = ModelCheckpoint(
                        monitor='val_loss',                # Monitor validation loss
                        dirpath=save_path,         # Directory to save checkpoints
                        filename='model-{epoch:02d}',  # Custom filename pattern
                        save_top_k=-1,                     # Save all checkpoints that are checked 
                        every_n_epochs = 10, # Number of epochs between checkpoints being cheked 
                        mode='min',                         # Minimize the monitored metric
                        enable_version_counter = False # Whether to append a version to the existing file name.
    )
    return checkpoint_callback

def early_stopping(patience):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=False, mode='min')
    return early_stopping


def s3l_pretrain(X_pretrain, continuous_cols, categorical_cols, pretrain_type, results_dir_name):
    
    if pretrain_type == 's3l_dae':
        pretrain_model, pre_trainer = s3l_pretrain_with_dae(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_dae, results_dir_name)
    elif pretrain_type == 's3l_vime':
        pretrain_model, pre_trainer = s3l_pretrain_with_vime(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_vime, results_dir_name)
    elif pretrain_type == 's3l_scarf':
        pretrain_model, pre_trainer = s3l_pretrain_with_scarf(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_scarf, results_dir_name)
    elif pretrain_type == 's3l_subtab':
        pretrain_model, pre_trainer = s3l_pretrain_with_subtab(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_subtab, results_dir_name) 
    elif pretrain_type == 's3l_switchtab':
        pretrain_model, pre_trainer = s3l_pretrain_with_switchtab(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_switchtab, results_dir_name) 
    return pretrain_model, pre_trainer


def s3l_pretrain_with_subtab(X_unlabeled, continuous_cols, categorical_cols, hypp, save_path):
    
    print('HYPERPARAMETERS: ', hypp)
    config = SubTabConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                          input_dim=X_unlabeled.shape[1], hidden_dim=hypp['hidden_dim'], 
                          output_dim=1, tau=hypp['tau'], use_cosine_similarity=hypp['use_cosine_similarity'], use_contrastive=hypp['use_contrastive'], 
                          use_distance=hypp['use_distance'], n_subsets=hypp['n_subsets'], overlap_ratio=hypp['overlap_ratio'],
                          mask_ratio=hypp['mask_ratio'], noise_type=hypp['noise_type'])

    pl_module = SubTabLightning(config)
    # create a train and validation set 
    X_unlabeled, X_train = train_test_split(X_unlabeled, test_size=0.05, shuffle=True)  
    X_unlabeled, X_valid = train_test_split(X_unlabeled, test_size=0.1, shuffle=True)
    
    ### First Phase Learning
    train_ds = SubTabDataset(X_train, unlabeled_data=X_unlabeled)
    valid_ds = SubTabDataset(X_valid)
    
    datamodule = TS3LDataModule(train_ds, valid_ds, hypp['batch_size'], train_sampler='random', 
                                train_collate_fn=SubTabCollateFN(config), valid_collate_fn=SubTabCollateFN(config), n_jobs = 5)
    checkpoint_callback = get_checkpoint(save_path)
    
    trainer = Trainer(logger=False, accelerator = 'gpu', max_epochs = hypp['max_epochs'], 
                      callbacks=[MyProgressBar(), checkpoint_callback])
    trainer.fit(pl_module, datamodule)
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss)

    return pl_module, trainer

def s3l_pretrain_with_switchtab(X_unlabeled, continuous_cols, categorical_cols, hypp, save_path):
    
    print('HYPERPARAMETERS: ', hypp)
    config = SwitchTabConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                             input_dim=X_unlabeled.shape[1], hidden_dim=hypp['hidden_dim'], output_dim=1, 
                             encoder_depth=hypp['encoder_depth'], n_head = hypp['n_head'], u_label = hypp['u_label'])

    pl_module = SwitchTabLightning(config)
    # create a train and validation set 
    X_unlabeled, X_train = train_test_split(X_unlabeled, test_size=0.05, shuffle=True)  
    X_unlabeled, X_valid = train_test_split(X_unlabeled, test_size=0.1, shuffle=True)
    ### First Phase Learning
    train_ds = SwitchTabDataset(X = X_train, unlabeled_data = X_unlabeled, config=config, continuous_cols=continuous_cols, category_cols=categorical_cols)
    valid_ds = SwitchTabDataset(X = X_valid, config=config, continuous_cols=continuous_cols, category_cols=categorical_cols)

    datamodule = TS3LDataModule(train_ds, valid_ds, hypp['batch_size'], train_sampler='random', 
                                train_collate_fn=SwitchTabFirstPhaseCollateFN(), valid_collate_fn=SwitchTabFirstPhaseCollateFN(), n_jobs = 5)
    
    checkpoint_callback = get_checkpoint(save_path)
    trainer = Trainer(logger=False, ccelerator = 'gpu', max_epochs = hypp['max_epochs'], 
                      callbacks=[MyProgressBar(), checkpoint_callback])
    trainer.fit(pl_module, datamodule)
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss)

    return pl_module, trainer

        
def s3l_pretrain_with_scarf(X_unlabeled, continuous_cols, categorical_cols, hypp, save_path):
    
    print('HYPERPARAMETERS: ', hypp)
    config = SCARFConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={},
                         input_dim=X_unlabeled.shape[1], hidden_dim=hypp['hidden_dim'],
                         output_dim=1, encoder_depth=hypp['encoder_depth'], head_depth=hypp['head_depth'],
                         dropout_rate=hypp['dropout_rate'], corruption_rate = hypp['corruption_rate']) 
    
    pl_module = SCARFLightning(config)
    # create a train and validation set 
    X_unlabeled, X_train = train_test_split(X_unlabeled, test_size=0.05, shuffle=True)  
    X_unlabeled, X_valid = train_test_split(X_unlabeled, test_size=0.1, shuffle=True)

    ### First Phase Learning
    train_ds = SCARFDataset(X = X_train, unlabeled_data=X_unlabeled, config = config)
    valid_ds = SCARFDataset(X = X_valid, config=config)
    
    datamodule = TS3LDataModule(train_ds, valid_ds, hypp['batch_size'], train_sampler='random', n_jobs = 5)
    checkpoint_callback = get_checkpoint(save_path)
    trainer = Trainer(logger=False, accelerator = 'gpu', max_epochs = hypp['max_epochs'], 
                      callbacks=[MyProgressBar(), checkpoint_callback])
    trainer.fit(pl_module, datamodule)
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss)

    return pl_module, trainer

def s3l_pretrain_with_vime(X_unlabeled, continuous_cols, categorical_cols, hypp, save_path):
    
    print('HYPERPARAMETERS: ', hypp)
    config = VIMEConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={},
        input_dim=X_unlabeled.shape[1], hidden_dim=hypp['hidden_dim'], output_dim=1, 
        encoder_depth = hypp['encoder_depth'], dropout_rate = hypp['dropout_rate'],
        alpha1=hypp['alpha1'], alpha2=hypp['alpha2'], 
        beta=hypp['beta'], K=hypp['K'], p_m = hypp['p_m'],
        num_categoricals=len(categorical_cols), num_continuous=len(continuous_cols))
    
    pl_module = VIMELightning(config)
    # create a train and validation set 
    X_unlabeled, X_train = train_test_split(X_unlabeled, test_size=0.05, shuffle=True)  
    X_unlabeled, X_valid = train_test_split(X_unlabeled, test_size=0.1, shuffle=True)
    ### First Phase Learning
    train_ds = VIMEDataset(X = X_train, unlabeled_data = X_unlabeled, config=config, continuous_cols = continuous_cols, category_cols = categorical_cols)
    valid_ds = VIMEDataset(X = X_valid, config=config, continuous_cols = continuous_cols, category_cols = categorical_cols)
    datamodule = TS3LDataModule(train_ds, valid_ds, hypp['batch_size'], train_sampler='random', n_jobs = 5)
    checkpoint_callback = get_checkpoint(save_path)
    trainer = Trainer(logger=False, accelerator = 'gpu', max_epochs = hypp['max_epochs'], 
                      callbacks=[MyProgressBar(), checkpoint_callback])
    trainer.fit(pl_module, datamodule)
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss)
    
    return pl_module, trainer


def s3l_pretrain_with_dae(X_unlabeled, continuous_cols, categorical_cols, hypp, save_path):
    
    print('HYPERPARAMETERS: ', hypp)
    config = DAEConfig (input_dim=X_unlabeled.shape[1],
                       hidden_dim=hypp['hidden_dim'], encoder_depth=hypp['encoder_depth'], 
                       noise_type = hypp['noise_type'], noise_ratio = hypp['noise_ratio'], 
                       num_categoricals=len(categorical_cols), num_continuous=len(continuous_cols),
                       # These are specific to tasks and not to pretraining
                       task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                       head_depth = hypp['head_depth'], output_dim=1
                      )
        
    pl_module = DAELightning(config)
    # Pretraining
    # create a train and validation set 
    X_unlabeled, X_train = train_test_split(X_unlabeled, test_size=0.05, shuffle=True)  
    X_unlabeled, X_valid = train_test_split(X_unlabeled, test_size=0.1, shuffle=True)
    train_ds = DAEDataset(X = X_train, unlabeled_data = X_unlabeled, continuous_cols = continuous_cols, category_cols = categorical_cols)
    valid_ds = DAEDataset(X = X_valid, continuous_cols = continuous_cols, category_cols = categorical_cols)
    datamodule = TS3LDataModule(train_ds, valid_ds, hypp['batch_size'], train_sampler='random', 
                                train_collate_fn=DAECollateFN(config), valid_collate_fn=DAECollateFN(config), n_jobs = 5) # made valid_ds None
    checkpoint_callback = get_checkpoint(save_path)
    trainer = Trainer(logger=False, accelerator = 'gpu', max_epochs = hypp['max_epochs'], 
                      callbacks=[MyProgressBar(), checkpoint_callback])
    trainer.fit(pl_module, datamodule)
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss)

    return pl_module, trainer


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



def s3l_load(load_path, type):
    
    assert (type in ['s3l_dae', 's3l_vime', 's3l_scarf', 's3l_subtab', 's3l_switchtab'], 
                        f"Invalid pretrain_model_to_load_type: {type}.")
    if type == 's3l_dae':
        model = DAELightning.load_from_checkpoint(load_path)
    elif type == 's3l_vime':
        model = VIMELightning.load_from_checkpoint(load_path)
    elif type == 's3l_scarf':
        model = SCARFLightning.load_from_checkpoint(load_path)
    elif type == 's3l_subtab':
        model = SubTabLightning.load_from_checkpoint(load_path)
    elif type == 's3l_switchtab':
        model = SwitchTabLightning.load_from_checkpoint(load_path)
    return model
    

def prepare_dataloaders(pretrain_model_to_load_type, 
                                        X_train, y_train,
                                        X_val, y_val,
                                        X_test,
                                        is_regression, continuous_cols, category_cols, 
                                        batch_size):
    if pretrain_model_to_load_type == 's3l_dae':
    
        train_ds = DAEDataset(X = X_train, Y = y_train, is_regression=is_regression,
                              continuous_cols=continuous_cols, category_cols=categorical_cols)
        valid_ds = DAEDataset(X = X_val, Y = y_val, is_regression=is_regression,
                              continuous_cols=continuous_cols, category_cols=categorical_cols)
        datamodule = TS3LDataModule(train_ds, valid_ds, is_regression=is_regression,
                                    batch_size = s3l_hyp_pred_head['batch_size'], train_sampler="random")
        # Evaluation
        test_ds = DAEDataset(X_test, category_cols=categorical_cols, 
                             continuous_cols=continuous_cols, is_regression=is_regression)
        test_dl = DataLoader(test_ds, s3l_hyp_pred_head['batch_size'], 
                             shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)
        train_ds = DAEDataset(X_train, continuous_cols=continuous_cols, 
                              category_cols=categorical_cols, is_regression=is_regression)
        train_dl = DataLoader(train_ds, s3l_hyp_pred_head['batch_size'], 
                             shuffle=False, sampler = SequentialSampler(train_ds), num_workers=4)
    
    elif pretrain_model_to_load_type == 's3l_vime':
        
        train_ds = VIMEDataset(X = X_train, Y = y_train, is_regression=is_regression,
                               continuous_cols=continuous_cols, category_cols=categorical_cols, 
                               is_second_phase=True)
        valid_ds = VIMEDataset(X = X_val, Y = y_val, is_regression=is_regression,
                               continuous_cols=continuous_cols, category_cols=categorical_cols, 
                               is_second_phase=True)   
        datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = s3l_hyp_pred_head['batch_size'], is_regression=is_regression,
                                    train_sampler="random", train_collate_fn=VIMESecondPhaseCollateFN()) 
        # Evaluation
        test_ds = VIMEDataset(X_test, category_cols=categorical_cols, is_regression=is_regression,
                             continuous_cols=continuous_cols, is_second_phase=True)
        test_dl = DataLoader(test_ds, s3l_hyp_pred_head['batch_size'], 
                             shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)
        train_ds = VIMEDataset(X_train, continuous_cols=continuous_cols, is_regression=is_regression, 
                              category_cols=categorical_cols, is_second_phase=True)
        train_dl = DataLoader(train_ds, s3l_hyp_pred_head['batch_size'], 
                             shuffle=False, sampler = SequentialSampler(train_ds), num_workers=4)
    
    elif pretrain_model_to_load_type == 's3l_scarf':
    
        train_ds = SCARFDataset(X_train, y_train, is_regression=is_regression, is_second_phase=True)
        valid_ds = SCARFDataset(X_val, y_val, is_regression=is_regression, is_second_phase=True)
        datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = s3l_hyp_pred_head['batch_size'], is_regression=is_regression, 
                                    train_sampler="random")
        # Evaluation
        test_ds = SCARFDataset(X_test, is_regression=is_regression, is_second_phase=True)
        test_dl = DataLoader(test_ds, s3l_hyp_pred_head['batch_size'], 
                             shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)
        train_ds = SCARFDataset(X_train, is_regression=is_regression, is_second_phase=True)
        train_dl = DataLoader(train_ds, s3l_hyp_pred_head['batch_size'], 
                             shuffle=False, sampler = SequentialSampler(train_ds), num_workers=4)
    
    elif pretrain_model_to_load_type == 's3l_subtab':
        
        # Only subtab seems to demand that config be passed into it again, so doing it here 
        
        config = SubTabConfig( task="regression", loss_fn="MSELoss", metric=s3l_hyp_ssl_subtab['metric'], metric_hparams={}, 
                              input_dim=X_train.shape[1], hidden_dim=s3l_hyp_ssl_subtab['hidden_dim'], output_dim=1, 
                              tau=s3l_hyp_ssl_subtab['tau'], use_cosine_similarity=s3l_hyp_ssl_subtab['use_cosine_similarity'], 
                              use_contrastive=s3l_hyp_ssl_subtab['use_contrastive'], use_distance=s3l_hyp_ssl_subtab['use_distance'], 
                              n_subsets=s3l_hyp_ssl_subtab['n_subsets'], overlap_ratio=s3l_hyp_ssl_subtab['overlap_ratio'], 
                              mask_ratio=s3l_hyp_ssl_subtab['mask_ratio'], noise_type=s3l_hyp_ssl_subtab['noise_type'])
        
        train_ds = SubTabDataset(X_train, y_train, is_regression=is_regression)
        valid_ds = SubTabDataset(X_val, y_val, is_regression=is_regression)
        datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = s3l_hyp_pred_head['batch_size'], is_regression=is_regression, 
                                    train_sampler="random", train_collate_fn=SubTabCollateFN(config), 
                                    valid_collate_fn=SubTabCollateFN(config))
        # Evaluation
        test_ds = SubTabDataset(X_test, is_regression=is_regression)
        test_dl = DataLoader(test_ds, s3l_hyp_pred_head['batch_size'], collate_fn=SubTabCollateFN(config),
                             shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)
        #test_dl = DataLoader(test_ds, s3l_hyp_pred_head['batch_size'],
        #                     shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)
        train_ds = SubTabDataset(X_train, is_regression=is_regression)
        train_dl = DataLoader(train_ds, s3l_hyp_pred_head['batch_size'], collate_fn=SubTabCollateFN(config),
                             shuffle=False, sampler = SequentialSampler(train_ds), num_workers=4)
        #train_dl = DataLoader(train_ds, s3l_hyp_pred_head['batch_size'],
        #                     shuffle=False, sampler = SequentialSampler(train_ds), num_workers=4)
    
    elif pretrain_model_to_load_type == 's3l_switchtab':
     
        train_ds = SwitchTabDataset(X_train, y_train, continuous_cols=continuous_cols, category_cols=categorical_cols, 
                                    is_regression=is_regression, is_second_phase=True)
        valid_ds = SwitchTabDataset(X_val, y_val, continuous_cols=continuous_cols, category_cols=categorical_cols, 
                                    is_regression=is_regression, is_second_phase=True)
        datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = s3l_hyp_pred_head['batch_size'], is_regression=is_regression, 
                                    train_sampler="random")
        # Evaluation
        test_ds = SwitchTabDataset(X_test, continuous_cols=continuous_cols, category_cols=categorical_cols, 
                                   is_regression=is_regression, is_second_phase=True)
        test_dl = DataLoader(test_ds, s3l_hyp_pred_head['batch_size'], 
                             shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)
        train_ds = SwitchTabDataset(X_train, continuous_cols=continuous_cols, category_cols=categorical_cols, 
                                    is_regression=is_regression, is_second_phase=True)
        train_dl = DataLoader(train_ds, s3l_hyp_pred_head['batch_size'], 
                             shuffle=False, sampler = SequentialSampler(train_ds), num_workers=4)

    return datamodule, train_dl, test_dl



#==================================================
# Scalers, Models and Training functions 
#==================================================

def identity_transform(X):
    return X

# We need to try different Scalers
def create_format_specific_scaler (train, categorical_cols, scaler_type):
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

    combined_scaler = ColumnTransformer(
    transformers=[
        ('cont_scaler', val_scaler, train.columns.difference(categorical_cols)),
        ('cat_pass', FunctionTransformer(identity_transform), categorical_cols)
    ],
    remainder='passthrough'
    )

    combined_scaler.fit(train)
    return combined_scaler


def create_scaler (train, categorical_cols, scaler_type):
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


def predict_from_dataloader(model: nn.Sequential, dataloader: DataLoader, model_type):
    #print('encoder model type is ', type(model))
    if model_type == 's3l_vime':
        model = model.encoder
        #print(summary(model, input_size=(1, 92)))
    elif model_type == 's3l_subtab':
        model = model.encoder
        #print(summary(model, input_size=(1, batch_size*s3l_hyp_ssl_subtab['n_subsets'])))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    
    all_predictions = []
    with torch.no_grad():  # Disable gradients for inference
        for batch in dataloader:
            if model_type == 's3l_dae':
                inputs = batch
            else: 
                # For subtab we have a tuple ([40,40], [batch_size, 92], [batch_size])
                # for the rest we have a list [[batch_size, 92], [batch_size]]
                inputs = batch[0]

            inputs = inputs.to(device)  # Move inputs to the appropriate device
            outputs = model(inputs)
            if model_type == 's3l_subtab':
                outputs = arrange_tensors(outputs, s3l_hyp_ssl_subtab['n_subsets'])
                outputs = outputs.reshape(inputs.shape[0] // s3l_hyp_ssl_subtab['n_subsets'], s3l_hyp_ssl_subtab['n_subsets'], -1).mean(1)

            # Collect predictions
            all_predictions.append(outputs.cpu())  # Move outputs to CPU if needed
    
    return torch.cat(all_predictions)


def get_xgb_model(X_train, y_train, X_val, y_val, model_to_save_name, hyper_params, learning_task_type):
    if learning_task_type == 'reg':
        if hyper_params['loss'][learning_task_type] == 'mae':
            model = XGBRegressor(**params, n_jobs=30, 
                                 learning_rate=hyper_params['learning_rate'], # Set the learning rate (0.1 is a common default)
                                 n_estimators=hyper_params['n_estimators'], # Number of trees (adjust based on your data size)
                                 max_depth=hyper_params['max_depth'] # Max depth of trees (control model complexity)
                                )
        elif hyper_params['loss'][learning_task_type] == 'mape':
            model = XGBRegressor(n_jobs=30, objective=mape_obj,
                                 learning_rate=hyper_params['learning_rate'], # Set the learning rate (0.1 is a common default)
                                 n_estimators=hyper_params['n_estimators'], # Number of trees (adjust based on your data size)
                                 max_depth=hyper_params['max_depth'] # Max depth of trees (control model complexity)
                                )
                            #eval_metric=mean_absolute_percentage_error,
        else:# use the default which is mse
            model = XGBRegressor(n_jobs=30,
                                 learning_rate=hyper_params['learning_rate'], # Set the learning rate (0.1 is a common default)
                                 n_estimators=hyper_params['n_estimators'], # Number of trees (adjust based on your data size)
                                 max_depth=hyper_params['max_depth'] # Max depth of trees (control model complexity)
                                )

    else: #'clas'
        model = XGBClassifier(n_jobs=30, use_label_encoder=False, 
                              learning_rate=hyper_params['learning_rate'], # Set the learning rate (0.1 is a common default)
                              n_estimators=hyper_params['n_estimators'], # Number of trees (adjust based on your data size)
                              max_depth=hyper_params['max_depth'] # Max depth of trees (control model complexity)
                             ) 
        # use_label_encoder=False added to remove a deprecation warning, no effect on performance
    
    # fit the model
    history = model.fit(X_train, y_train)
    model.save_model(model_to_save_name+'.json')
    
    return model, history

def get_pytorch_mlp_model(X_train, y_train, X_val, y_val, model_to_save_name, hyper_params, learning_task_type):    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                            torch.tensor(y_train, dtype=torch.float32)), 
                              batch_size=hyper_params['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                            torch.tensor(y_val, dtype=torch.float32)), 
                              batch_size=hyper_params['batch_size'], shuffle=True)
    
    if learning_task_type == 'reg':
        # Model instantiation
        print('Blue BLue You are bLue ')
        print('loss_fn_str: ', hyper_params['loss']['reg'])
        model = MLPRegressor(input_dim=X_train.shape[1], 
                             hidden_dims=hyper_params['fc_layers'], output_dim=1, 
                             use_batchnorm=hyper_params['use_batchnorm'], use_dropout=hyper_params['use_dropout'], 
                             dropout_rate=hyper_params['dropout_rate'], 
                             learning_rate=hyper_params['learning_rate'], loss_fn_str=hyper_params['loss']['reg'])

    elif learning_task_type == 'clas':
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        # Model instantiation
        model = MLPClassifier(input_dim=X_train.shape[1],
                             hidden_dims=hyper_params['fc_layers'], output_dim=y_train.shape[1],
                             use_batchnorm=True, use_dropout=True, 
                             dropout=hyper_params['dropout_rate'], learning_rate=hyper_params['learning_rate'], loss_fn_str=hyper_params['loss']['clas'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_losses', patience=hyper_params['patience'], 
                                   mode='min', verbose=True)
    
    # Model checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_losses',        # Monitors validation loss
        dirpath='checkpoints/',    # Directory to save checkpoints
        filename='best-checkpoint', # Checkpoint filename
        save_top_k=1,              # Only save the best model
        mode='min',                 # Minimize the validation loss
        enable_version_counter = False # Whether to append a version to the existing file name.
    )
    
    trainer = pl.Trainer(logger=False, accelerator = 'gpu',
                         max_epochs=hyper_params['max_epochs'],
                         num_sanity_val_steps = 0,
                         callbacks=[MyProgressBar(refresh_rate=10)] # early_stopping, checkpoint_callback, 
                         #default_root_dir=suptrain_models_folder+suptrain_model_to_save_name
                        )
    trainer.fit(model, train_loader, val_loader)
    print(ModelSummary(model, max_depth=-1))
    
    # Load the best checkpoint after training
    #best_model_path = checkpoint_callback.best_model_path
    #best_model = MLPRegressor.load_from_checkpoint(best_model_path)
    #best_model.save(model_to_save_name)
    # Now `best_model` has the weights from the best validation epoch

    print(len(model.train_losses))
    print(len(model.val_losses))
    print(len(model.val_mapes))
    history = {'train_losses': model.train_losses,
               'val_losses': model.val_losses,
               'val_mapes': model.val_mapes}

    return model, history

    
def get_mlp_model(X_train, y_train, X_val, y_val, model_to_save_name, hyper_params, learning_task_type):
    
    if learning_task_type == 'clas':
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        
    model = Sequential ()
    for l in range(0,len(hyper_params['fc_layers'])):
        # Hidden layers
        if l==0:
            model.add(Dense(hyper_params['fc_layers'][l], input_dim=X_train.shape[1], 
                            kernel_initializer='he_uniform', activation='relu'))
        else:
            model.add(BatchNormalization())
            model.add(Dropout(0.1))
            model.add(Dense(hyper_params['fc_layers'][l], kernel_initializer='he_uniform', activation='relu'))
            
    model.add(Dense(y_train.shape[1], activation = hyper_params['out_activation'][learning_task_type]))
    optimizer = Adam_tf(learning_rate=hyper_params['learning_rate'])
    #optimizer = (learning_rate=hyper_params['learning_rate'])
    
    model.compile(loss=hyper_params['loss'][learning_task_type], optimizer=optimizer, metrics=hyper_params['metrics'][learning_task_type])

    # Fit the model 
    print_every_n_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(f"Epoch: {epoch + 1}, Loss: {logs['loss']}") if (epoch + 1) % 10 == 0 else None)
    early_stopping = EarlyStopping(monitor='val_loss', patience=hyper_params['patience'], restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        epochs=hyper_params['max_epochs'], 
                        batch_size=hyper_params['batch_size'], 
                        validation_data=(X_val, y_val), 
                        callbacks=[early_stopping, print_every_n_callback], 
                        shuffle=True, 
                        verbose=0)

    model.save(model_to_save_name)
    
    return model, history

def get_tabnet_model(X_train, y_train, X_val, y_val, model_to_save_name, hyper_params, learning_task_type):
    if learning_task_type == 'reg':
        model = TabNetRegressor()
    else:
        model = TabNetClassifier()
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

def train_model (X_train, X_val, y_train, y_val, sup_model_type, learning_task_type, model_to_save_name, hyper_params, sample_weights=None, save_str=[]):
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    
    if sup_model_type == 'mlp': 
        model, history = get_pytorch_mlp_model(X_train, y_train, X_val, y_val, model_to_save_name, hyper_params, learning_task_type)
        plot_model_train_info (history['train_losses'], history['val_losses'])
    elif sup_model_type == 'xgb':
        model, history = get_xgb_model(X_train, y_train, X_val, y_val, model_to_save_name, hyper_params, learning_task_type)
    elif sup_model_type == 'tabnet':
        model, history = get_tabnet_model(X_train, y_train, X_val, y_val, model_to_save_name, hyper_params, learning_task_type)
    else:    
        print('Do not know model')

    return model 


#==========================================
# Plotting functions
#==========================================

def plot_model_train_info (train_losses, val_losses):
    plt.figure(1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.show()
    #plt.savefig(model_save_path + 'loss_curve_' + '.pdf')

    return True

def plot_y_yhat (y_train,yhat_train,y_test,yhat_test, model_save_path):
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
    #plt.savefig(model_save_path + 'train_y_yhat'
    #             + '.pdf')
    
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
    #plt.savefig(model_save_path + 'test_y_yhat'
    #             + '.pdf')
    
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
    #print(X_feats[np.flip(indices)[0:feat_filter]])

    #top_n_features = list( set(top_n_features).union(set(bar_names)))
    #print('Top n feature list: ', top_n_features)
    plt.figure(figsize=(7, 4))
    plt.barh(range(len(bar_vals)), np.flip(bar_vals), color='b', align='center')
    plt.yticks(range(len(bar_vals)), np.flip(bar_names))

    plt.title('Feature importance')
    plt.xlabel('Relative Importance')
    #plt.savefig('plots_for_paper/feat_imp'+'.pdf', bbox_inches='tight')
    plt.show() 
    return

def draw_confusion_matrix(true, pred, title):                
    cm = confusion_matrix(true, pred, normalize='true')
    # Create a confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=np.unique(true), yticklabels=np.unique(pred))
    # Add labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    # Show the plot
    plt.show()

# Function to set up each subplot for the Q-Q plots
def setup_axes(ax, x, y, title, color, bounds):
    ax.plot(x, y, color, marker='.', linestyle='none')
    ax.set_title(title)
    ax.set_xlabel('Ground truth')
    ax.set_ylabel('Predictions')
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.plot(bounds, bounds, 'k-')  # Diagonal line
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    # Set ticks at fixed intervals (modify interval as needed)
    tick_locator = MaxNLocator(nbins=5)  # Adjust number of bins as needed
    ax.xaxis.set_major_locator(tick_locator)
    ax.yaxis.set_major_locator(tick_locator)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    ax.set_xticklabels(ax.get_xticks().round(3), fontsize=14)
    ax.set_yticklabels(ax.get_yticks().round(3), fontsize=14)
    return

# Plot hist of output
def plot_hist_of_y(y_train, y_test, learning_task):
    # If classification it will just bin it
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4))
    fig.subplots_adjust(top=0.85, bottom=0.20)
    fig.supxlabel(learning_task)
    fig.suptitle('Histogram')
    ax1.hist(y_train, bins=50, color='r', edgecolor='k', label='train samples')
    #ax1.set_xlabel(learning_task)
    ax1.set_yticks([])
    ax1.set_title('Train')
    ax2.hist(y_test, bins=50, color='b', edgecolor='k', label='test_samples')
    
    ax2.set_yticks([])
    ax2.set_title('Test')
    plt.show()

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
        optimizer=Adam(lr=IN_PARAM['learning_rate']),
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
