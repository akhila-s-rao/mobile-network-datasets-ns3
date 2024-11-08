
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
from plotting_functions import *

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


def make_data_pretrain_ready (pretrain_data: pd.DataFrame, learning_tasks, all_learning_tasks_in_data, unlab_dataset_scenario='pd1'):

    print('pretrain_data, before removing rows that dont have traffic ', pretrain_data.shape)

    if unlab_dataset_scenario == 'pd1':
        # Results in 1.8 M samples
        traffic_cols = pretrain_data.columns
        pretrain_data = pretrain_data.dropna(subset=traffic_cols, how='all', axis=0)
    elif unlab_dataset_scenario == 'pd2':
        # Results in 33K samples
        traffic_cols = list(set(learning_tasks) - {'delay_trace.txt_ul_delay', 'delay_trace.txt_dl_delay'})
        pretrain_data = pretrain_data.dropna(subset=traffic_cols, how='all', axis=0)
    elif unlab_dataset_scenario == 'pd3':   
        # Results in 102K samples 
        traffic_cols = list(set(all_learning_tasks_in_data) - {'delay_trace.txt_ul_delay', 'delay_trace.txt_dl_delay'})
        pretrain_data = pretrain_data.dropna(subset=traffic_cols, how='all', axis=0)
    elif unlab_dataset_scenario == 'pd4':
        traffic_cols = list(set(all_learning_tasks_in_data) - {'delay_trace.txt_ul_delay', 'delay_trace.txt_dl_delay'}) 
        pretrain_data_traffic = pretrain_data.dropna(subset=traffic_cols, how='all', axis=0)
        pretrain_data_quiet = pretrain_data[pretrain_data[traffic_cols].isna().all(axis=1)].sample(n=pretrain_data_traffic.shape[0], random_state=561)
        pretrain_data = pd.concat([pretrain_data_traffic, pretrain_data_quiet], axis=0, ignore_index=True)

    print('pretrain_data, after removing rows that dont have traffic ', pretrain_data.shape)
    
    # Remove the labels of all prediction tasks which are also in the dataset 
    X_pretrain = pretrain_data.drop(all_learning_tasks_in_data, axis=1, errors='ignore')
    # Drop rows that have NaNs 
    X_pretrain = X_pretrain.dropna()
    print('X_pretrain ', X_pretrain.shape)
    
    return X_pretrain # pandas

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
        print('HYPERPARAMETERS: ', hypp)
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
        print('HYPERPARAMETERS: ', hypp)
        config = VIMEConfig(task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                            input_dim=input_dim, hidden_dim=hypp['hidden_dim'], output_dim=1, 
                            encoder_depth = hypp['encoder_depth'], dropout_rate = hypp['dropout_rate'],
                            alpha1=hypp['alpha1'], alpha2=hypp['alpha2'], 
                            beta=hypp['beta'], K=hypp['K'], p_m = hypp['p_m'],
                            num_categoricals=len(categorical_cols), num_continuous=len(continuous_cols))
        model = VIMELightning(config)
    elif type == 's3l_scarf':
        hypp = s3l_hyp_ssl_scarf
        print('HYPERPARAMETERS: ', hypp)
        config = SCARFConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={},
                             input_dim=input_dim, hidden_dim=hypp['hidden_dim'],
                             output_dim=1, encoder_depth=hypp['encoder_depth'], head_depth=hypp['head_depth'],
                             dropout_rate=hypp['dropout_rate'], corruption_rate = hypp['corruption_rate']) 
        model = SCARFLightning(config)
    elif type == 's3l_subtab':
        hypp = s3l_hyp_ssl_subtab
        print('HYPERPARAMETERS: ', hypp)
        config = SubTabConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                              input_dim=input_dim, hidden_dim=hypp['hidden_dim'], 
                              output_dim=1, tau=hypp['tau'], use_cosine_similarity=hypp['use_cosine_similarity'], use_contrastive=hypp['use_contrastive'], 
                              use_distance=hypp['use_distance'], n_subsets=hypp['n_subsets'], overlap_ratio=hypp['overlap_ratio'],
                              mask_ratio=hypp['mask_ratio'], noise_type=hypp['noise_type'])
        model = SubTabLightning(config)
    elif type == 's3l_switchtab':
        hypp = s3l_hyp_ssl_switchtab
        print('HYPERPARAMETERS: ', hypp)
        config = SwitchTabConfig( task="regression", loss_fn=hypp['loss_fn'], metric=hypp['metric'], metric_hparams={}, 
                                     input_dim=input_dim, hidden_dim=hypp['hidden_dim'], output_dim=1, 
                                     encoder_depth=hypp['encoder_depth'], n_head = hypp['n_head'], u_label = hypp['u_label'])
        model = SwitchTabLightning(config)
    return model


#========================================
# Pretraining functions
#========================================
# Used during first phase pretraining
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

# Used during second phase finetuning
def get_best_r2_checkpoint (save_path):
    checkpoint_callback = ModelCheckpoint(
            monitor='val_r2_score',        # Monitors validation r2 score 
            dirpath=save_path,    # Directory to save checkpoints
            filename='{epoch}-best-r2-{val_r2_score:.3f}', # Checkpoint filename
            save_top_k=1,              # Only save the best model
            mode='max',                 # Maximize the validation r2 score
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
    plt_str='loss_curve_subtab.pdf'
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss, 'plots/'+plt_str)

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
    trainer = Trainer(logger=False, accelerator = 'gpu', max_epochs = hypp['max_epochs'], 
                      callbacks=[MyProgressBar(), checkpoint_callback])
    trainer.fit(pl_module, datamodule)
    plt_str='loss_curve_switchtab.pdf'
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss, 'plots/'+plt_str)

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
    plt_str='loss_curve_scarf.pdf'
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss, 'plots/'+plt_str)

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
    plt_str='loss_curve_vime.pdf'
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss, 'plots/'+plt_str)
    
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
    plt_str='loss_curve_dae.pdf'
    plot_model_train_info (pl_module.first_phase_train_loss, pl_module.first_phase_val_loss, 'plots/'+plt_str)

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
                        batch_size,
                        X_pretrain = None):
    
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
        
        hypp=s3l_hyp_ssl_vime
        config = VIMEConfig( task="regression", input_dim=X_train.shape[1], output_dim=1, loss_fn=hypp['loss_fn'],
                            alpha1=hypp['alpha1'], alpha2=hypp['alpha2'], 
                            beta=hypp['beta'], K=hypp['K'], p_m = hypp['p_m'], num_categoricals=1) # num_categoricals wont be used here anyway so I dont care what I send 
        
        train_ds = VIMEDataset(X = X_train, Y = y_train, 
                               config=config, 
                               unlabeled_data=X_pretrain, 
                               is_regression=is_regression, continuous_cols=continuous_cols, category_cols=categorical_cols, is_second_phase=True)
        valid_ds = VIMEDataset(X = X_val, Y = y_val, config=config, is_regression=is_regression,
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
        plt_str='loss_curve_switchtab.pdf'
        plot_model_train_info (history['train_losses'], history['val_losses'], 'plots/'+plt_str)
    elif sup_model_type == 'xgb':
        model, history = get_xgb_model(X_train, y_train, X_val, y_val, model_to_save_name, hyper_params, learning_task_type)
    elif sup_model_type == 'tabnet':
        model, history = get_tabnet_model(X_train, y_train, X_val, y_val, model_to_save_name, hyper_params, learning_task_type)
    else:    
        print('Do not know model')

    return model 