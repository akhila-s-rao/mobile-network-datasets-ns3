
import warnings
import logging
import os

# DEBUG MODE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
    
from helper_functions import *




# This includes first and second ohase training
def s3l_training(iteration, pt_type, pretrain, use_pretrained_model, freeze_encoder):
    # Sets the random seed
    # If you want to set a specific seed then pass it as an argument 
    initialize(561)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    notebook_save_str = 's3l_'+pt_type
    
    #========================================
    # Experiment Parameters: Check carefully!
    #========================================
    time_step_size = '500ms'
    # This is where models are saved and loaded from
    #dataset_folder = '../../../dataset_ver1/parsed_data_'+time_step_size+'_5steps/' # 500 ms with 5 100ms steps 
    dataset_folder = '../../../../dataset_ver1/parsed_data_'+time_step_size+'_singleStep/'
    #dataset_folder = '../../../dataset_ver1/parsed_data_'+time_step_size+'_singleStep/'
    pretrain_models_folder = 'models/'+pt_type+'_pretrain/'
    suptrain_models_folder = iteration+'/'
    pretrain_slice = 'all' #['macro', 'micro', 'slow', 'fast', 'all']
    train_slice = 'all' #['macro', 'micro', 'slow', 'fast', 'all', 'only_delay']
    test_slice = 'all' #['macro', 'micro', 'slow', 'fast', 'all']
    EXP_PARAM = {
        'scaler': 'standard', #'minmax', 'standard', 'robust', 'maxabs', 'l2norm'
        'num_rand_runs': 3 # number of runs with each run doing a different random sample of size label_no from the set of labeled samples 
    }
    
    num_samples_list = [100, 1*K, 10*K, 20*K]
    #num_samples_list = [10*K]
    #==================================
    # Pretraining Experiment Parameters
    #==================================

    #pretrain = False
    pretrain_type = 's3l_'+pt_type
    pretrain_model_to_save_name = pretrain_slice+'_'+pretrain_type
    
    scaler_to_save_name = pretrain_type+'_'+EXP_PARAM['scaler']+'_scaler'
    
    #==========================================
    # Supervised Training Experiment Parameters
    #==========================================
    
    # Load an existing pretrained model to use as encoding for sup model 
    #use_pretrained_model = True
    pretrain_model_to_load_type = 's3l_'+pt_type
    pretrain_model_to_load_name = pretrain_slice+'_'+pretrain_type
    scaler_to_load_name = pretrain_type+'_'+EXP_PARAM['scaler']+'_scaler.pkl'
    
    # Train a sup model with or without using a pretrained model 
    sup_model_type = 'mlp' # xgb
    suptrain_model_to_save_name = iteration+'_'+train_slice+'_'+pretrain_model_to_load_type+'_'+sup_model_type # could also be sup_model_with_pretrain 
    
    
    #==================================================
    # Experiment Parameters: Not often changed
    #==================================================
    
    # When input features are NA 
    # Could experiment with forward fill imputation 
    # If the label is NA during supervised training then the sample is dropped  
    impute_method = 'forward_fill'# ['forward_fill', 'zero_fill']
                                  
    # These are the ones we have chosen to work with  
    #learning_tasks = ['httpClientRtt_trace.txt_page_load_time']
    learning_tasks = [#'dashClient_trace.txt_newBitRate_bps', 
                      'vrFragment_trace.txt_vr_frag_thput_mbps', 'vrFragment_trace.txt_vr_burst_thput_mbps',
                      'vrFragment_trace.txt_vr_frag_time', 'vrFragment_trace.txt_vr_burst_time', 
                      #'httpClientRtt_trace.txt_page_load_time',
                      'delay_trace.txt_ul_delay', 'delay_trace.txt_dl_delay']
    # index matched with the learning_tasks above
    #learning_task_types = ['reg']
    learning_task_types = [#'clas', 
                           'reg', 'reg', 
                           'reg', 'reg',
                           #'reg',
                           'reg', 'reg']
    
    # If True then we are predicting one window ahead if False then we are predicting on the same window 
    shift_samp_for_predict = True
    
    # If you want the test samples to be sorted by delay value to see the error differences for the low delay and high delay cases 
    sort_test_samples = False
    
    use_all_feats = True
    # take the top n features of each run and add it to the top_n_features list  
    # If use_all_feats = True then thes will not be used 
    feat_filter = 10 
    top_n_features = []
    # Only valid when use_all_feats = False 
    selected_features = []
    
    # All delay values above this will be clipped to the threshold value
    clip_outliers = True
    delay_clip_th = 5000 # ms
    
    
    
    # Hyperparameters for SSL using DAE 
    #print('WARNING !!! HYPERPARAMETERS IN DEBUG MODE')
    # Hyperparameters for supervised XGB training 
    hypp_sup_xgb={
        'loss':{'reg':'mse',
                'clas':'categorical_crossentropy'}
    } 
    
    # Hyperparameters for supervised MLP training
    hypp_sup_mlp={
        'fc_layers': [200], # the hidden layers
        'batch_size': 64,  
        'max_epochs': 100,
        'patience': 15,
        'learning_rate': 0.0001,
        'use_batchnorm': False,
        'use_dropout': False,
        'dropout_rate': 0.1,
        'loss':{'reg':'mse',#'mean_absolute_error' 'mean_absolute_percentage_error'
                'clas':'categorical_crossentropy'},
        'metrics':{
                   'reg':['MeanAbsolutePercentageError'],
                   'clas':['Recall']},
        'out_activation':{'reg':'linear', 
                          'clas':'softmax'} 
    }
    
    s3l_hyp_pred_head={
        'batch_size': 32,
        'max_epochs': 30,
        'patience': 5,
        'loss':{'reg':'MSELoss',
                'clas':'CrossEntropyLoss'},
        'metrics':{'reg':['MeanAbsolutePercentageError'],
                   'clas':['Recall']},#'F1Score' does not work needs a different dimension 
    }
    
    s3l_hyp_ssl_dae={
        'metric': "mean_absolute_percentage_error", #
        'hidden_dim': 200, #
        'max_epochs': 100, #
        'batch_size': 128, #
        
        'encoder_depth': 4, #  
        'head_depth': 2, #
        'dropout_rate': 0.1, #
        
        'noise_type': "Swap", #
        'noise_ratio': 0.3, #
    }
    
    s3l_hyp_ssl_vime={
        'metric': "mean_absolute_percentage_error", #
        'hidden_dim': 200, #
        'max_epochs': 100, #
        'batch_size': 128, #

        # NO ENCODER DEPTH 
        # NO HEAD DEPTH
        # NO DROPOUT
        
        'p_m': 0.3, # Corruption probability for self-supervised learning
        
        'alpha1': 2.0, # Hyper-parameter to control the weights of feature and mask losses
        'alpha2': 2.0, # Hyper-parameter to control the weights of feature and mask losses
        'K': 3, # Number of augmented samples
        'beta': 1.0 # Hyperparameter to control supervised and unsupervised losses
    }
    
    s3l_hyp_ssl_scarf={
        'metric': "mean_absolute_percentage_error",
        'hidden_dim': 200,
        'max_epochs': 100,
        'batch_size': 128,
        
        'encoder_depth': 4,
        'head_depth': 2,
        'dropout_rate': 0.1, #
        
        'corruption_rate': 0.3
    }
    
    s3l_hyp_ssl_subtab={
        'metric': "mean_absolute_percentage_error",
        'hidden_dim': 200,
        'max_epochs': 100,
        'batch_size': 128,
        
        'encoder_depth': 4,
        'head_depth': 2,
        # NO DROPOUT

        'noise_type': "Swap",
        'noise_level': 0.3,
        
        'tau': 1.0,
        'use_cosine_similarity': True,
        'use_contrastive': True,
        'use_distance': True,
        'n_subsets': 4,
        'overlap_ratio': 0.75,
        'mask_ratio': 0.1
        
        
    }
    
    s3l_hyp_ssl_switchtab={
        'metric': "mean_absolute_percentage_error",
        'hidden_dim': 200,
        'max_epochs': 100,
        'batch_size': 128,
        
        'encoder_depth': 4,
        # NO HEAD DEPTH
        'dropout_rate': 0.1, #
        
        'n_head': 2,
        'u_label': -1
    }



    
    # Hyperparameters for SSL using TabNet
    s3l_hyp_ssl_tabnet={
        #'lambda_sparse': , # default = 1e-3
        # This is the extra sparsity loss coefficient as proposed in the original paper. 
        # The bigger this coefficient is, the sparser your model will be in terms of feature selection. 
        # Depending on the difficulty of your problem, reducing this value could help.
        'mask_type': 'entmax', # 'entmax' # default='sparsemax'
        'n_da': 8, # between 8-64 # default=8
        'n_steps': 3, # between 3-10 # default=3
        'n_independent': 2, # between 1-5 # default=2
        'n_shared': 2, # between 1-5 # default=2
        'n_shared_decoder': 1, # default=1
        'n_indep_decoder': 1, # default=1
        'noise_ratio': 0.30,
        'batch_size': 1024, #default=1024
        'max_epochs': 30, # default=200
        'patience': 15 # default=10
    }
    
    
    if sup_model_type == 'mlp':
        sup_hyper_params=hypp_sup_mlp
    elif sup_model_type == 'xgb':
        sup_hyper_params=hypp_sup_xgb    
    
    
    #pretrain_runs = range(1, 10 + 1)
    #train_runs = range(11, 17 + 1)
    #test_runs = range(18, 20 + 1)
    
    num_pretrain_runs = 10 # 10
    num_train_runs = 7 # 7 
    num_test_runs = 3 # 3
    train_test_run_nums = np.array(range(11, 20+1))
    #train_test_run_nums = np.array(range(11, 12+1))
    

    pretrain_runs = range(1, 10 + 1)
    #pretrain_runs = range(1, 1 + 1)
    
    
    # Create the pretrain_models_folder if it does not already exist
    if not os.path.isdir(pretrain_models_folder):
        os.makedirs(pretrain_models_folder)

    if pretrain: 
        # Read the pretraining data
        pretrain_data = read_and_concatenate_runs(pretrain_runs, dataset_folder, 
                                                  pretrain_slice, network_info, time_step_size, 
                                                  use_all_feats, drop_col_substr, learning_tasks, 
                                                  shift_samp_for_predict, 
                                                  impute_method, sum_cols_substr, all_learning_tasks_in_data)
        
        # Remove the labels of all prediction tasks which are also in the dataset 
        X_pretrain = pretrain_data.drop(all_learning_tasks_in_data, axis=1, errors='ignore')
        X_pretrain = X_pretrain.dropna()
        X_cols = X_pretrain.columns
        continuous_cols, categorical_cols = get_cont_and_cat_cols(X_pretrain)
        print(X_pretrain.shape)
        # Create, save and use scaler
        val_scaler = create_scaler(X_pretrain, EXP_PARAM['scaler'])
        with open(pretrain_models_folder + scaler_to_save_name +'.pkl', 'wb') as f:
            pickle.dump(val_scaler, f)
        
        # transform returns a numpy even when you pass a pandas dataframe 
        X_pretrain = val_scaler.transform(X_pretrain).copy()
        X_pretrain = pd.DataFrame(X_pretrain, columns=X_cols)
    
        assert (pretrain_type in ['s3l_dae', 's3l_vime', 's3l_scarf', 's3l_subtab', 's3l_switchtab'], 
                            f"Invalid pretrain_type: {pretrain_type}.")
        
        if pretrain_type == 's3l_dae':
            pretrain_model, pre_trainer = s3l_pretrain_with_dae(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_dae, 
                                                        pretrain_models_folder+pretrain_model_to_save_name)
        elif pretrain_type == 's3l_vime':
            pretrain_model, pre_trainer = s3l_pretrain_with_vime(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_vime, 
                                                        pretrain_models_folder+pretrain_model_to_save_name)
        elif pretrain_type == 's3l_scarf':
            pretrain_model, pre_trainer = s3l_pretrain_with_scarf(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_scarf, 
                                                        pretrain_models_folder+pretrain_model_to_save_name)
        elif pretrain_type == 's3l_subtab':
            pretrain_model, pre_trainer = s3l_pretrain_with_subtab(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_subtab, 
                                                        pretrain_models_folder+pretrain_model_to_save_name)
        elif pretrain_type == 's3l_switchtab':
            pretrain_model, pre_trainer = s3l_pretrain_with_switchtab(X_pretrain, continuous_cols, categorical_cols, s3l_hyp_ssl_switchtab, 
                                                        pretrain_models_folder+pretrain_model_to_save_name)
        
        # save this pretrained model for later use
        pre_trainer.save_checkpoint(pretrain_models_folder+pretrain_model_to_save_name+'.ckpt')
        print('DONE SAVING PRETRAINED MODEL')
        
        print(ModelSummary(pretrain_model, max_depth=-1))
        
        # If I just pretrained then I am done here 
        # Come back with pretrain set to false to do second phase training 
        print('DONE PRETRAINING')
        return





    # =======================
    # SECOND PHASE LEARNING
    # =======================
    
    # Create the supervised_models_folder if it does not already exist
    if not os.path.isdir(suptrain_models_folder):
        os.makedirs(suptrain_models_folder)
        
    # This section of the code should only be reached if weare NOT pretraining 
    train_results_filepath = iteration+'/'+notebook_save_str+'_train_results.csv'
    test_results_filepath = iteration+'/'+notebook_save_str+'_test_results.csv'

    # Erase the contents of the results file into which we append as we loop through 
    open(train_results_filepath, 'w').close()
    open(test_results_filepath, 'w').close()
 
    # Iterate over different number of labeled samples 
    for label_no in num_samples_list:
        
        EXP_PARAM['label_no'] = label_no
    
        # Aggregate results over runs
        train_results = pd.DataFrame(index=learning_tasks, columns=['MAE', 'MAPE', 'R2']) # or ['ACC', 'F1', 'ROC_AUC']
        test_results = pd.DataFrame(index=learning_tasks, columns=['MAE', 'MAPE', 'R2']) # or ['ACC', 'F1', 'ROC_AUC']
        # Apply the function to each cell in the DataFrame
        train_results = train_results.applymap(lambda x: create_nan_array(EXP_PARAM['num_rand_runs']))
        test_results = test_results.applymap(lambda x: create_nan_array(EXP_PARAM['num_rand_runs']))
    
        # Iterate over different random initializations
        for rs in range(0, EXP_PARAM['num_rand_runs']):
            read_data_start_time = time.time()
            #print('==========================================================================')
            #print('NUM. LABELLED SAMPLES: ', label_no)
            #print('==========================================================================')
            #print('==========================================================================')
            #print('Random iteration: ', rs)
            #print('==========================================================================')
                        
            # Randomly select which runs shall be train and which shall be test 
            shuffled_run_ind = np.random.permutation(len(train_test_run_nums))
            train_runs = train_test_run_nums[shuffled_run_ind[:num_train_runs]]
            test_runs = train_test_run_nums[shuffled_run_ind[num_train_runs:]]
            print('# train runs used ', len(train_runs))
            print('# test runs used ', len(test_runs))
            
            # Read the runs from files and creater a train and test dataset 
            train_data = read_and_concatenate_runs (train_runs, dataset_folder, train_slice, network_info, time_step_size, 
                                                   use_all_feats, drop_col_substr, learning_tasks, shift_samp_for_predict, 
                                                   impute_method, sum_cols_substr, all_learning_tasks_in_data)
            test_data = read_and_concatenate_runs (test_runs, dataset_folder, test_slice, network_info, time_step_size, 
                                                  use_all_feats, drop_col_substr, learning_tasks, shift_samp_for_predict, 
                                                  impute_method, sum_cols_substr, all_learning_tasks_in_data)
            print('Time to read data: ', time.time() - read_data_start_time)
            for idx, learning_task in enumerate(learning_tasks):
                
                task_it_start_time = time.time()
                print('==========================================================================')
                print('NUM. LABELLED SAMPLES: ', label_no)
                print('==========================================================================')
                print('==========================================================================')
                print('Random iteration: ', rs)
                print('==========================================================================')
                print('==========================================================================')
                print('Learning task: ', learning_task, ' task type: ', learning_task_types[idx], 
                      ' task num: ', idx, '/', len(learning_task_types))
                print('==========================================================================')
        
                learning_task_type = learning_task_types[idx]
                is_regression = learning_task_type == 'reg'
                learning_task_type_ff = 'regression'
                if learning_task_type == 'clas':
                    learning_task_type_ff = 'classification'
    
                X_train, y_train, train_strat_array, X_feats, continuous_cols, categorical_cols = make_data_sup_model_ready (train_data, learning_task, learning_task_type,
                                                                                         all_learning_tasks_in_data, bitrate_levels, 
                                                                                         clip_outliers, delay_clip_th)
                X_test, y_test, test_strat_array, _ , _, _= make_data_sup_model_ready (test_data, learning_task, learning_task_type, 
                                                                 all_learning_tasks_in_data, bitrate_levels, 
                                                                 clip_outliers, delay_clip_th)
                
                # Sample a subset of the train samples to be equal to the num of labelled samples we need
                sample_size = EXP_PARAM['label_no']
                if len(y_train) <= EXP_PARAM['label_no']:
                    print("\n\nWARNING !!!!\n\nAsked to sample from train set of size "+str(len(y_train))+ 
                          " a number greater than or equal to its size "+str(EXP_PARAM['label_no']))
                    print('Going to just take as many samples as available, No need to random sample')
                else:
                    # This functions helps me random sample with stratification 
                    X_train, _, y_train, _ = train_test_split(X_train, y_train,
                                                                          train_size=EXP_PARAM['label_no'], shuffle=True,
                                                                          stratify=train_strat_array)
                    
                # create a validation set from the test set 
                X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                                test_size=int(min(X_train.shape[0], 0.25*X_test.shape[0])),    
                                                                shuffle=True, stratify=test_strat_array)  
                
                print('Train data shape ' + str(X_train.shape))
                print('Test data shape ' + str(X_test.shape))
                print('Val data shape ' + str(X_val.shape))
        
                #plot_hist_of_y(y_train, y_test, learning_task)
                 
                print('Time to process data for one learning task: ', time.time() - task_it_start_time)
                
                #=============================================== Train and test the model ==================================
            
                start_time = time.time()
        
                if use_pretrained_model:
                    
                    assert pretrain_model_to_load_type in ['s3l_dae', 's3l_vime', 's3l_scarf', 's3l_subtab', 's3l_switchtab'], f"Invalid pretrain_model_to_load_type: {pretrain_model_to_load_type}."
                    # get the latent representations from pretrained model
                    # Load the pretrained model and scaler 
                    # Load the saved MinMaxScaler object from the file
                    with open(pretrain_models_folder + scaler_to_load_name, 'rb') as f:
                        val_scaler = pickle.load(f)

                    print('Loaded the scaler')
                    X_train = val_scaler.transform(X_train).copy()
                    X_test = val_scaler.transform(X_test).copy()
                    X_val = val_scaler.transform(X_val).copy()
                    # transform returns numpy. I need to convert back to pandas dataframe
                    X_train = pd.DataFrame(X_train, columns=X_feats)
                    X_val = pd.DataFrame(X_val, columns=X_feats)
                    X_test = pd.DataFrame(X_test, columns=X_feats)
                    
                    # load the model
                    model = s3l_load(pretrain_models_folder+pretrain_model_to_load_name+'.ckpt', pretrain_model_to_load_type)
                    # model is of type class 'ts3l.pl_modules.dae_lightning.DAELightning'
                    print(ModelSummary(model, max_depth=-1))
                    print('Loaded the pretrained model')
                    # set model to second phase
                    model.set_second_phase(freeze_encoder=freeze_encoder)
                    # prepare a trainer 
                    early_stopping = EarlyStopping(monitor='val_loss', patience=s3l_hyp_pred_head['patience'], verbose=False, mode='min')
                    model_checkpoint = ModelCheckpoint(
                        monitor='val_loss',   # Metric to monitor
                        save_top_k=1,         # Save only the best model
                        mode='min',           # Mode should be 'min' for metrics that should decrease
                        filename=notebook_save_str+'_{epoch}-{val_loss:.2f}',  # Format of saved files
                        save_last=True        # Save the last model
                    )
                    # Mode should be 'min' for metrics that should decrease
                    sup_trainer = Trainer(accelerator = 'gpu', 
                                          #devices = 'auto',
                                          max_epochs = s3l_hyp_pred_head['max_epochs'], 
                                          num_sanity_val_steps = 2, 
                                          log_every_n_steps=50,
                                          #early_stopping, 
                                          callbacks=[MyProgressBar(), model_checkpoint],
                                          default_root_dir=suptrain_models_folder+suptrain_model_to_save_name)
                    # Need to reset DAE config for every learning task 
    
    
                    print('Setup the Trainer for training using labeled samples over the pretrained model')
    
                    if pretrain_model_to_load_type == 's3l_dae':
     
                        train_ds = DAEDataset(X = X_train, Y = y_train, is_regression=is_regression,
                                              continuous_cols=continuous_cols, category_cols=categorical_cols)
                        valid_ds = DAEDataset(X = X_val, Y = y_val, is_regression=is_regression,
                                              continuous_cols=continuous_cols, category_cols=categorical_cols)
                        datamodule = TS3LDataModule(train_ds, valid_ds, is_regression=is_regression,
                                                    batch_size = s3l_hyp_pred_head['batch_size'], train_sampler="random")

                        print('Setup DataModule for the Trainer')
                        
                        sup_trainer.fit(model, datamodule)
                        #print(ModelSummary(model, max_depth=-1))
                        print('Done fine tuning pretrained model with labeled data')
                        # Load the best model 
                        #best_model_path = model_checkpoint.best_model_path
                        #print(f"Best model path: {best_model_path}")
                        #model = DAELightning.load_from_checkpoint(best_model_path)
                        
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
                        
                        sup_trainer.fit(model, datamodule)
                        #print(ModelSummary(model, max_depth=-1))
                        # Load the best model 
                        #best_model_path = model_checkpoint.best_model_path
                        #print(f"Best model path: {best_model_path}")
                        #model = VIMELightning.load_from_checkpoint(best_model_path)
    
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
                        
                        sup_trainer.fit(model, datamodule)
                        #print(ModelSummary(model, max_depth=-1))
                        # Load the best model 
                        #best_model_path = model_checkpoint.best_model_path
                        #print(f"Best model path: {best_model_path}")
                        #model = SCARFLightning.load_from_checkpoint(best_model_path)
    
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
                                              mask_ratio=s3l_hyp_ssl_subtab['mask_ratio'], noise_type=s3l_hyp_ssl_subtab['noise_type'], 
                                              noise_level=s3l_hyp_ssl_subtab['noise_level'])
                        
                        train_ds = SubTabDataset(X_train, y_train, is_regression=is_regression)
                        valid_ds = SubTabDataset(X_val, y_val, is_regression=is_regression)
                        
                        datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = s3l_hyp_pred_head['batch_size'], is_regression=is_regression, 
                                                    train_sampler="random", train_collate_fn=SubTabCollateFN(config), 
                                                    valid_collate_fn=SubTabCollateFN(config))
                        #datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = s3l_hyp_pred_head['batch_size'], is_regression=is_regression, 
                        #                            train_sampler="random")
                        
                        sup_trainer.fit(model, datamodule)
                        #print(ModelSummary(model, max_depth=-1))
                        # Load the best model 
                        #best_model_path = model_checkpoint.best_model_path
                        #print(f"Best model path: {best_model_path}")
                        #model = SubTabLightning.load_from_checkpoint(best_model_path)
    
                        # Evaluation
                        test_ds = SubTabDataset(X_test, is_regression=is_regression)
                        test_dl = DataLoader(test_ds, s3l_hyp_pred_head['batch_size'], collate_fn=SubTabCollateFN(config),
                                             shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)
                        train_ds = SubTabDataset(X_train, is_regression=is_regression)
                        train_dl = DataLoader(train_ds, s3l_hyp_pred_head['batch_size'], collate_fn=SubTabCollateFN(config),
                                             shuffle=False, sampler = SequentialSampler(train_ds), num_workers=4)
    
                    elif pretrain_model_to_load_type == 's3l_switchtab':
                     
                        train_ds = SwitchTabDataset(X_train, y_train, continuous_cols=continuous_cols, category_cols=categorical_cols, 
                                                    is_regression=is_regression, is_second_phase=True)
                        valid_ds = SwitchTabDataset(X_val, y_val, continuous_cols=continuous_cols, category_cols=categorical_cols, 
                                                    is_regression=is_regression, is_second_phase=True)
                        
                        datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = s3l_hyp_pred_head['batch_size'], is_regression=is_regression, 
                                                    train_sampler="random")
                        
                        sup_trainer.fit(model, datamodule)
                        #print(ModelSummary(model, max_depth=-1))
                        # Load the best model 
                        #best_model_path = model_checkpoint.best_model_path
                        #print(f"Best model path: {best_model_path}")
                        #model = SwitchTabLightning.load_from_checkpoint(best_model_path)
    
                        # Evaluation
                        test_ds = SwitchTabDataset(X_test, continuous_cols=continuous_cols, category_cols=categorical_cols, 
                                                   is_regression=is_regression, is_second_phase=True)
                        test_dl = DataLoader(test_ds, s3l_hyp_pred_head['batch_size'], 
                                             shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)
                        train_ds = SwitchTabDataset(X_train, continuous_cols=continuous_cols, category_cols=categorical_cols, 
                                                    is_regression=is_regression, is_second_phase=True)
                        train_dl = DataLoader(train_ds, s3l_hyp_pred_head['batch_size'], 
                                             shuffle=False, sampler = SequentialSampler(train_ds), num_workers=4)
    
    
    
    
                        
                    if learning_task_type == 'reg':
                        yhat_test = sup_trainer.predict(model, test_dl)
                        yhat_test = torch.concat([out.cpu() for out in yhat_test]).squeeze()
                        yhat_train = sup_trainer.predict(model, train_dl)
                        yhat_train = torch.concat([out.cpu() for out in yhat_train]).squeeze()
                        #print(y_train)
                        #print(yhat_train)
                        # Clip predictions to the range of the training data
                        print('NOTE: Clipping the predictions to be within the range of the ground-truth values')
                        yhat_train = np.clip(yhat_train, y_train.min(), y_train.max())
                        yhat_test = np.clip(yhat_test, y_train.min(), y_train.max())
                        #print(yhat_train)
                         
                    else: #'clas'
                        yhat_train_a = sup_trainer.predict(model, train_dl)
                        yhat_train_proba = F.softmax(torch.concat([out.cpu() for out in yhat_train_a]).squeeze(),dim=1)
                        yhat_train = yhat_train_proba.argmax(1)
                        yhat_test_a = sup_trainer.predict(model, test_dl)
                        yhat_test_proba = F.softmax(torch.concat([out.cpu() for out in yhat_test_a]).squeeze(),dim=1)
                        yhat_test = yhat_test_proba.argmax(1)
                        print(y_train)
                        print(yhat_train)
                        print(yhat_train_proba)        
                #accuracy = accuracy_score(y_test, yhat_test_proba.argmax(1))
                    
                else: # only supervised training 
                    val_scaler = create_scaler(X_train, EXP_PARAM['scaler'])
                    # save the scaler model to a file
                    with open(suptrain_models_folder + scaler_to_save_name + '.pkl', 'wb') as f:
                        pickle.dump(val_scaler, f)
                    
                    X_train = val_scaler.transform(X_train).copy()
                    X_test = val_scaler.transform(X_test).copy()
                    X_val = val_scaler.transform(X_val).copy()
                    # Need to train the model
                    # Train and save the model       
                    model = train_model(X_train, X_val, 
                                                y_train, y_val, 
                                                sup_model_type, learning_task_type,
                                                suptrain_models_folder + suptrain_model_to_save_name,
                                                sup_hyper_params)
                        
                    if learning_task_type == 'reg':
                        # Make predictions
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        # Set to inference mode. This disables regularization. 
                        model.eval()
                        
                        with torch.no_grad():
                            yhat_train = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
                            yhat_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

                        #yhat_test = model.predict(X_test).flatten()
                        #yhat_train = model.predict(X_train).flatten()
                        # Clip predictions to the range of the training data
                        print('NOTE: Clipping the predictions to be within the range of the ground-truth values')
                        yhat_train = np.clip(yhat_train, y_train.min(), y_train.max())
                        yhat_test = np.clip(yhat_test, y_train.min(), y_train.max())
                        
                    else: #learning_task_type == 'clas':
                        if sup_model_type == 'xgb':
                            yhat_test = model.predict(X_test).flatten()
                            yhat_train = model.predict(X_train).flatten()
                            yhat_test_proba = model.predict_proba(X_test)
                            yhat_train_proba = model.predict_proba(X_train)
                        else:
                            yhat_test = np.argmax(model.predict(X_test), axis=1)
                            yhat_train = np.argmax(model.predict(X_train), axis=1)
                            yhat_test_proba = model.predict(X_test)
                            yhat_train_proba = model.predict(X_train)
                            
                
                    end_time = time.time()
                    print('Time to train model: ', end_time - start_time)
                    if not use_pretrained_model:
                        # Feature importance
                        if ((sup_model_type == 'tabnet') or (sup_model_type == 'xgb')):
                            plot_feature_importance(model.feature_importances_, X_feats, feat_filter)
                
                #====================================
                # Compute error using the supervised model 
                #====================================
                
                if learning_task_type == 'reg':
                    if np.isnan(yhat_train).any():
                        print('WARNING: nans in the predicted train vals')
                        print(yhat_train)
                    train_results.loc[learning_task, 'MAE'][rs] = compute_error(y_train, yhat_train, 'mae')
                    train_results.loc[learning_task, 'MAPE'][rs] = compute_error(y_train, yhat_train, 'mape')
                    train_results.loc[learning_task, 'R2'][rs] = compute_error(y_train, yhat_train, 'r2')
                    # Compute and Print Test set errors
                    if np.isnan(yhat_test).any():
                        print('WARNING: NaNs in the predicted test vals')
                        print(yhat_test)
                    test_results.loc[learning_task, 'MAE'][rs] = compute_error(y_test, yhat_test, 'mae')
                    test_results.loc[learning_task, 'MAPE'][rs] = compute_error(y_test, yhat_test, 'mape')
                    test_results.loc[learning_task, 'R2'][rs] = compute_error(y_test, yhat_test, 'r2')
    
                    # Predicted versus ground-truth plots
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
                    fig.suptitle(learning_task, fontsize=16)
                    y = np.concatenate((y_train, y_test))
                    yhat = np.concatenate((yhat_train, yhat_test))
                    bounds=[min( min(y),min(yhat) ), max( max(y),max(yhat) )]        
                    # Configure each subplot
                    setup_axes(axes[0], y_train, yhat_train, 'Train', COLOUR_HEX[0], bounds)  # Adjust color as needed
                    setup_axes(axes[1], y_test, yhat_test, 'Test', COLOUR_HEX[1], bounds)  # Adjust color as needed
                    #plt.tight_layout()
                    plt.show()
                else: # clas
                    if np.isnan(yhat_train).any():
                        print('WARNING: nans in the predicted train vals')
                        print(yhat_train)                
                    
                    train_results.loc[learning_task, 'MAE'][rs] = compute_error(y_train, yhat_train, 'acc')
                    train_results.loc[learning_task, 'MAPE'][rs] = compute_error(y_train, yhat_train, 'f1score')
                    train_results.loc[learning_task, 'R2'][rs] = compute_error(y_train, yhat_train_proba, 'roc_auc')
                
                    # Compute and Print Test set errors
                    if np.isnan(yhat_test).any():
                        print('WARNING: NaNs in the predicted test vals')
                        print(yhat_test)
                    test_results.loc[learning_task, 'MAE'][rs] = compute_error(y_test, yhat_test, 'acc')
                    test_results.loc[learning_task, 'MAPE'][rs] = compute_error(y_test, yhat_test, 'f1score')
                    test_results.loc[learning_task, 'R2'][rs] = compute_error(y_test, yhat_test_proba, 'roc_auc')
                    
                    # Confusion matrix plots
                    draw_confusion_matrix(y_train, yhat_train, 'Train: Confusion Matrix')
                    draw_confusion_matrix(y_test, yhat_test, 'Test: Confusion Matrix')
               
                print('')
                print(test_results)
                print('')
                print('===============================  DONE  ===================================================')
                
        print('============================= Train set mean over runs ===================================')
        mean_train_results = train_results.applymap(mean_array)
        print(mean_train_results)
        print('=============================== Test set results =========================================')
        print(test_results.applymap(format_array))
        print('============================= Test set mean over runs ====================================')
        mean_test_results = test_results.applymap(mean_array)
        print(test_results.applymap(mean_array))
        print('==========================================================================================')
        # Add number of samples as a column
        #mean_train_results.insert(0, 'num_samples', EXP_PARAM['label_no'])
        #mean_test_results.insert(0, 'num_samples', EXP_PARAM['label_no'])
        # Append to file
        mean_train_results.round(3).to_csv(train_results_filepath, mode='a', header=False)    
        mean_test_results.round(3).to_csv(test_results_filepath, mode='a', header=False)

    
    os.system('cp second_phase_training.py '+suptrain_models_folder+notebook_save_str+'.py')
    os.system('cp '+train_results_filepath+' '+suptrain_models_folder+'.')
    os.system('cp '+test_results_filepath+' '+suptrain_models_folder+'.')



# call it directly
if __name__ == '__main__':
    second_phase_training(sys.argv[1])


#from torch.multiprocessing import Process
## Run this parallelly for all the models
#if __name__ == '__main__':
#    model_types = ['dae']#['dae', 'vime', 'scarf', 'subtab']
#    processes = []
#    print(torch.cuda.is_available())
#    print(torch.version.cuda) 
#    for model_type in model_types:
#        p = Process(target=second_phase_training, args=(model_type,))
#        p.start()
#        processes.append(p)

#    for p in processes:
#        p.join()


    
#from torch.multiprocessing import Pool
#if __name__ == '__main__':
#    pool = Pool(processes=4)
#    inputs = [(type,) for type in ['dae', 'vime']]
#    #inputs = [(type,) for type in ['dae']]
#    outputs = pool.starmap(second_phase_training, inputs)  
#    #outputs = pool.map(setup_run, range(0,runs))
#    print(outputs)
