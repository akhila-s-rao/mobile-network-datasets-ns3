
import warnings
import logging
import os

# DEBUG MODE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
    
from helper_functions import *
from hyperparameters import *
from compute_plot_save_error import *
from plotting_functions import *
from pytorch_mlp import *

# This includes first and second ohase training
def s3l_training(pretrain, use_pretrained_model, 
                 pred_head_type = 'ts3l',# 'ts3l', 'xgb'
                 pt_type='',
                 unlab_dataset_scenario = 'pd1', # 'pd1' - 'pd4'
                 sup_model_type = '',
                 # Baseline
                 random_initialize_model=False,
                 # if mlp 
                 sup_mlp_layers = None,
                 SP_results_folder='', pt_folder='', 
                 freeze_encoder=True, pred_head_size=None, 
                 shift_samp_for_predict = True # If True then we are predicting one window ahead if False then we are predicting on the same window
                ):

    #==========================
    # INITIALIZE
    #==========================
    
    first_pass = True
    # Sets the random seed 
    initialize(561)
    #initialize(420)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    notebook_save_str = 's3l_'+pt_type
    
    #========================================
    # Experiment Parameters: Check carefully!
    #========================================
    time_step_size = '500ms'
    # This is where models are saved and loaded from
    #dataset_folder = '../../../dataset_ver1/parsed_data_'+time_step_size+'_5steps/' # 500 ms with 5 100ms steps 
    dataset_folder = '../../../dataset_ver1/parsed_data_'+time_step_size+'_singleStep/'
    pretrain_models_folder = 'models/'+pt_folder+'/'+pt_type+'_pretrain/'
    suptrain_models_folder = SP_results_folder+'/'
    pretrain_slice = 'all' #['macro', 'micro', 'slow', 'fast', 'all']
    train_slice = 'all' #['macro', 'micro', 'slow', 'fast', 'all', 'only_delay']
    test_slice = 'all' #['macro', 'micro', 'slow', 'fast', 'all']
    EXP_PARAM = {
        'scaler': 'standard', #'minmax', 'standard', 'robust', 'maxabs', 'l2norm'
        'num_rand_runs': 1 # number of runs with each run doing a different random sample of size label_no from the set of labeled samples 
    }
    
    #num_samples_list = [100, 1*K, 10*K, 20*K]
    #num_samples_list = [100, 250, 500, 1*K, 10*K, 20*K]
    num_samples_list = [1000, 20*K]
    
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
    pretrain_model_to_load_type = 's3l_'+pt_type
    pretrain_model_to_load_name = pretrain_slice+'_'+pretrain_type
    scaler_to_load_name = pretrain_type+'_'+EXP_PARAM['scaler']+'_scaler.pkl'
    
    # Train a sup model with or without using a pretrained model 
    # could also be sup_model_with_pretrain
    suptrain_model_to_save_name = SP_results_folder+'_'+train_slice+'_'+pretrain_model_to_load_type+'_'+sup_model_type  
    
    #==================================================
    # Experiment Parameters: Not often changed
    #==================================================
    
    # When input features are NA 
    # Could experiment with forward fill imputation 
    # If the label is NA during supervised training then the sample is dropped  
    impute_method = 'forward_fill'# ['forward_fill', 'zero_fill']
                                  
    # These are the ones we have chosen to work with  
    learning_tasks = [#'dashClient_trace.txt_newBitRate_bps', 
                      'vrFragment_trace.txt_vr_frag_thput_mbps', 'vrFragment_trace.txt_vr_burst_thput_mbps',
                      'vrFragment_trace.txt_vr_frag_time', 'vrFragment_trace.txt_vr_burst_time', 
                      #'httpClientRtt_trace.txt_page_load_time',
                      #'delay_trace.txt_ul_delay', 
                      'delay_trace.txt_dl_delay']
    # index matched with the learning_tasks above
    learning_task_types = [#'clas', 
                           'reg', 'reg', 
                           'reg', 'reg',
                           #'reg',
                           #'reg', 
                           'reg'] 
    
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
    
    #num_train_runs = 7 # 7 
    #num_test_runs = 3 # 3
    #train_test_run_nums = np.array(range(11, 20+1))
    num_train_runs = 1 # 7 
    num_test_runs = 1 # 3
    train_test_run_nums = np.array(range(11, 12+1))
    
    #pretrain_runs = range(1, 10 + 1)
    pretrain_runs = range(1, 1 + 1)
    
    # Create the pretrain_models_folder if it does not already exist
    if not os.path.isdir(pretrain_models_folder):
        os.makedirs(pretrain_models_folder)

    #==========================
    # TRAIN SSL MODEL
    #==========================
    
    if pretrain: 
        # Read the pretraining data
        pretrain_data = read_and_concatenate_runs(pretrain_runs, dataset_folder, 
                                                  pretrain_slice, network_info, time_step_size, 
                                                  use_all_feats, drop_col_substr, learning_tasks, 
                                                  shift_samp_for_predict, 
                                                  impute_method, sum_cols_substr, all_learning_tasks_in_data)
        
        X_pretrain = make_data_pretrain_ready (pretrain_data, learning_tasks, all_learning_tasks_in_data, unlab_dataset_scenario)
        
        X_cols = X_pretrain.columns
        continuous_cols = list(set(X_cols) - set(categorical_cols))
    
        # Create a scaler and save it
        # X_pretrain should be a pandas dataframe 
        val_scaler = create_format_specific_scaler(X_pretrain, categorical_cols, EXP_PARAM['scaler'])
        with open(pretrain_models_folder + scaler_to_save_name +'.pkl', 'wb') as f:
            pickle.dump(val_scaler, f)
        
        # transform returns a numpy even when you pass a pandas dataframe 
        X_pretrain = pd.DataFrame(val_scaler.transform(X_pretrain).copy(), columns=X_cols)
        
        # Is this needed ? # Check 
        #X_pretrain[categorical_cols] = X_pretrain[categorical_cols].astype(int)
    
        # Pretrain
        start_time = time.time()
        pretrain_model, pre_trainer = s3l_pretrain(X_pretrain, continuous_cols, categorical_cols, pretrain_type, 
                                                        pretrain_models_folder+pretrain_model_to_save_name)

        # save this pretrained model for later use
        pre_trainer.save_checkpoint(pretrain_models_folder+pretrain_model_to_save_name+'.ckpt')
        print('DONE SAVING PRETRAINED MODEL')
        print(ModelSummary(pretrain_model, max_depth=-1))
        
        
        # If I just pretrained then I am done here 
        # Come back with pretrain set to false to do second phase training
        print('Time to pretrain model: ', time.time() - start_time)
        print('DONE PRETRAINING')

        return







    # ===========================
    # LEARNING FROM LABELED DATA
    # ===========================


    # ===========================
    # INITIALIZE
    # ===========================
    
    shuffled_run_ind = {}
    # Randomly select which runs shall be train and which shall be test for each random instance of the model run
    for rs in range(0, EXP_PARAM['num_rand_runs']):
        shuffled_run_ind[rs] = np.random.permutation(len(train_test_run_nums))
        train_runs = train_test_run_nums[shuffled_run_ind[rs][:num_train_runs]]
        test_runs = train_test_run_nums[shuffled_run_ind[rs][num_train_runs:]]
        print('rand_itr: ', rs, ' train_runs: ', train_runs, ' test_runs: ', test_runs)

    # Create the supervised_models_folder if it does not already exist
    if not os.path.isdir(suptrain_models_folder):
        os.makedirs(suptrain_models_folder)
        
    # This section of the code should only be reached if we are NOT pretraining 
    train_results_filepath = SP_results_folder+'/'+notebook_save_str+'_train_results.csv'
    test_results_filepath = SP_results_folder+'/'+notebook_save_str+'_test_results.csv'

    # Erase the contents of the results file into which we append as we loop through 
    open(train_results_filepath, 'w').close()
    open(test_results_filepath, 'w').close()

    # VIME needs unlabeled data for it's semi supervised learning in SP
    if pretrain_model_to_load_type == 's3l_vime':
        # Read the pretraining data
        pretrain_data = read_and_concatenate_runs(pretrain_runs, dataset_folder, 
                                                  pretrain_slice, network_info, time_step_size, 
                                                  use_all_feats, drop_col_substr, learning_tasks, 
                                                  shift_samp_for_predict, 
                                                  impute_method, sum_cols_substr, all_learning_tasks_in_data)
        
        X_pretrain = make_data_pretrain_ready (pretrain_data, learning_tasks, all_learning_tasks_in_data, unlab_dataset_scenario)
        X_cols = X_pretrain.columns        
        # Load the saved Scaler object from the file
        with open(pretrain_models_folder + scaler_to_load_name, 'rb') as f:
            val_scaler = pickle.load(f)
        # Transform
        X_pretrain = pd.DataFrame(val_scaler.transform(X_pretrain).copy(), columns=X_cols)
    else:
        X_pretrain = None
        
    
    # =================================================
    # LOOP OVER DIFFERENT NUMBER OF LABELED SAMPLES
    # =================================================
    script_runtime = time.time()
    
    for label_no in num_samples_list:
        
        num_samples_runtime = time.time()
        
        EXP_PARAM['label_no'] = label_no
    
        # Aggregate results over runs
        train_results = pd.DataFrame(index=learning_tasks, columns=['MAE', 'MAPE', 'R2']) # or ['ACC', 'F1', 'ROC_AUC']
        test_results = pd.DataFrame(index=learning_tasks, columns=['MAE', 'MAPE', 'R2']) # or ['ACC', 'F1', 'ROC_AUC']
        # Apply the function to each cell in the DataFrame
        train_results = train_results.applymap(lambda x: create_nan_array(EXP_PARAM['num_rand_runs']))
        test_results = test_results.applymap(lambda x: create_nan_array(EXP_PARAM['num_rand_runs']))

        # =================================================
        # LOOP OVER DIFFERENT RANDOM INITIALIZATIONS
        # =================================================

        for rs in range(0, EXP_PARAM['num_rand_runs']):
            
            rand_itr_runtime = time.time()
            
            read_data_start_time = time.time()
                        
            # Randomly select which runs shall be train and which shall be test
            train_runs = train_test_run_nums[shuffled_run_ind[rs][:num_train_runs]]
            test_runs = train_test_run_nums[shuffled_run_ind[rs][num_train_runs:]]
            #train_runs = [1]
            #test_runs = [2]
            
            print('train runs used ', train_runs)
            print('test runs used ', test_runs)
            
            # Read the runs from files and creater a train and test dataset 
            train_data = read_and_concatenate_runs (train_runs, dataset_folder, train_slice, network_info, time_step_size, 
                                                   use_all_feats, drop_col_substr, learning_tasks, shift_samp_for_predict, 
                                                   impute_method, sum_cols_substr, all_learning_tasks_in_data)
            test_data = read_and_concatenate_runs (test_runs, dataset_folder, test_slice, network_info, time_step_size, 
                                                  use_all_feats, drop_col_substr, learning_tasks, shift_samp_for_predict, 
                                                  impute_method, sum_cols_substr, all_learning_tasks_in_data)
            print('Time to read data: ', time.time() - read_data_start_time)
            
            # =================================================
            # LOOP OVER ALL LEARNING TASKS
            # =================================================
            
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
                assert (learning_task_type in ['reg', 'clas'], 
                                            f"Invalid learning_task_type: {learning_task_type}.")
                is_regression = learning_task_type == 'reg'
                
                learning_task_type_ff = 'regression'
                if learning_task_type == 'clas':
                    learning_task_type_ff = 'classification'
                
                X_train, y_train, train_strat_array, X_feats = make_data_sup_model_ready (train_data, learning_task, learning_task_type,
                                                                                         all_learning_tasks_in_data, bitrate_levels, 
                                                                                         clip_outliers, delay_clip_th)
                X_test, y_test, test_strat_array, _ = make_data_sup_model_ready (test_data, learning_task, learning_task_type, 
                                                                 all_learning_tasks_in_data, bitrate_levels, 
                                                                 clip_outliers, delay_clip_th)

                continuous_cols = list(set(X_feats) - set(categorical_cols))
                
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
                # X_train and X_test are pandas dataframes here 
                #=============================================== Train and test the model ==================================
            
                start_time = time.time()                    
                    
                # =================================================
                # USE SSL MODEL OR USE RANDOMLY INITIALIZED SSL MODEL
                # =================================================
                if use_pretrained_model or random_initialize_model:
                    
                    #==========================
                    # INITIALIZE
                    #==========================
                    
                    assert (pretrain_model_to_load_type in ['s3l_dae', 's3l_vime', 's3l_scarf', 's3l_subtab', 's3l_switchtab'], 
                            f"Invalid pretrain_model_to_load_type: {pretrain_model_to_load_type}.")
                    assert (pred_head_type in ['xgb', 'ts3l'], 
                            f"Invalid pred_head_type: {pred_head_type}.")
                    if pred_head_type == 'ts3l':
                        assert (pred_head_size in [1, 2], 
                            f"Invalid pred_head_size: {pred_head_size}.")
                    
                    if use_pretrained_model: 
                        # Load the saved Scaler object from the file
                        with open(pretrain_models_folder + scaler_to_load_name, 'rb') as f:
                            val_scaler = pickle.load(f)
                    else: #random_initialize_model
                        # Create a Scaler object 
                        val_scaler = create_format_specific_scaler(X_train, categorical_cols, EXP_PARAM['scaler'])
        
                    X_train = val_scaler.transform(X_train).copy()
                    X_test = val_scaler.transform(X_test).copy()
                    X_val = val_scaler.transform(X_val).copy()
                                        
                    # transform returns numpy. I need to convert back to pandas dataframe becasue the Ts3l library needs it in this format 
                    X_train = pd.DataFrame(X_train, columns=X_feats)
                    X_val = pd.DataFrame(X_val, columns=X_feats)
                    X_test = pd.DataFrame(X_test, columns=X_feats)                                                
                    
                    # I am setting batch size based on the number of samples in the train set 
                    if EXP_PARAM['label_no'] <= 5000:
                        s3l_hyp_pred_head['batch_size'] = 10 
                        sup_hyper_params=hypp_sup_xgb_small    
                            
                    else: # when it is greater
                        s3l_hyp_pred_head['batch_size'] = 200
                        sup_hyper_params=hypp_sup_xgb_large
                        
                    if pretrain_model_to_load_type == 's3l_vime':
                            s3l_hyp_pred_head['batch_size'] = 2000 

                    if use_pretrained_model: 
                        # load the model
                        ssl_model = s3l_load(pretrain_models_folder+pretrain_model_to_load_name+'.ckpt', pretrain_model_to_load_type)
                        if first_pass:
                            print(ModelSummary(ssl_model, max_depth=-1))
                            print('HYPERPARAMETERS: ', s3l_hyp_pred_head)
                            first_pass = False
                    else: #random_initialize_model
                        # Create a new model that is randomly initialized
                        ssl_model = initialize_s3l_model_random(type=pretrain_model_to_load_type, input_dim=X_train.shape[1], 
                                                    categorical_cols=categorical_cols, continuous_cols=continuous_cols)
                        if first_pass:
                            print(ModelSummary(ssl_model, max_depth=-1))
                            print('HYPERPARAMETERS: ', s3l_hyp_pred_head)
                            first_pass = False
                        

                    # Create the datamodeule needed for second phase training and the dataloaders needed for evaluation  
                    datamodule, train_dl, test_dl = prepare_dataloaders(pretrain_model_to_load_type, 
                                        X_train, y_train,
                                        X_val, y_val,
                                        X_test,
                                        is_regression, continuous_cols=continuous_cols, category_cols=categorical_cols, 
                                        batch_size = s3l_hyp_pred_head['batch_size'],
                                        X_pretrain=X_pretrain)
                    
                    
                    #============================
                    # TRAIN AND MAKE PREDICTIONS
                    #============================
                    
                    # set model to second phase
                    ssl_model.set_second_phase(freeze_encoder=freeze_encoder, pred_head_size=pred_head_size)
                    
                    # =================================================
                    # USE XGB PREDICTION HEAD 
                    # =================================================
                    if pred_head_type == 'xgb':

                        # Extract encoding from the first phase without using a dataloader
                        #encoder_model = ssl_model.model.encoder
                        #X_train_en = encoder_model(X_train.to_numpy()).detach().numpy()
                        #X_val_en = None
                        #X_test_en = encoder_model(X_test.to_numpy()).detach().numpy()
                        
                        # Extract encoding from the first phase
                        encoder_model = ssl_model.model.encoder
                        X_train_en = predict_from_dataloader(encoder_model, train_dl, 
                                                             pretrain_model_to_load_type).detach().numpy()
                        X_val_en = None
                        X_test_en = predict_from_dataloader(encoder_model, test_dl, 
                                                            pretrain_model_to_load_type).detach().numpy()
                        
                        print('X_train_en ', X_train_en.shape)
                        print('X_test_en', X_test_en.shape)
                        
                        # Use this as input to an XGB model 
                        model = train_model(X_train_en, X_val_en, 
                            y_train, y_val, 
                            pred_head_type, learning_task_type,
                            suptrain_models_folder + suptrain_model_to_save_name,
                            sup_hyper_params)
                        
                        if learning_task_type == 'reg':   
                            yhat_train = model.predict(X_train_en).flatten()
                            yhat_test = model.predict(X_test_en).flatten()
                            # Clip predictions to the range of the training data
                            print('NOTE: Clipping the predictions to be within the range of the ground-truth values')
                            yhat_train = np.clip(yhat_train, y_train.min(), y_train.max())
                            yhat_test = np.clip(yhat_test, y_train.min(), y_train.max())
                            
                        else: #learning_task_type == 'clas':
                            yhat_test = model.predict(X_test_en).flatten()
                            yhat_train = model.predict(X_train_en).flatten()
                            yhat_test_proba = model.predict_proba(X_test_en)
                            yhat_train_proba = model.predict_proba(X_train_en)

                    # =================================================
                    # USE TS3L INBUILT PREDICTION HEAD 
                    # =================================================
                    elif pred_head_type == 'ts3l':
                        
                        # prepare a trainer 
                        best_ckpt_save_path = suptrain_models_folder+suptrain_model_to_save_name
                        model_checkpoint = get_best_r2_checkpoint(best_ckpt_save_path)
                        print(best_ckpt_save_path)
                        sup_trainer = Trainer(logger=False, accelerator = 'gpu',
                                              max_epochs = s3l_hyp_pred_head['max_epochs'], 
                                              enable_progress_bar=True,
                                              callbacks=[MyProgressBar(), model_checkpoint]
                                             )
                        # Train
                        sup_trainer.fit(ssl_model, datamodule)
                        # Plot the validation metric 
                        plot_model_val_info(ssl_model.second_phase_val_metric)
                        
                        # Load the best model 
                        print(f"Best model path: {model_checkpoint.best_model_path}")
                        best_ssl_model = s3l_load(model_checkpoint.best_model_path, pretrain_model_to_load_type)
                        # Set the best model loaded from checkpoint to second phase
                        best_ssl_model.set_second_phase(freeze_encoder=freeze_encoder, pred_head_size=pred_head_size)
                        ssl_model = best_ssl_model
                        print('Loading best r2 score checkpoint')

                        if learning_task_type == 'reg':
                            yhat_test = sup_trainer.predict(ssl_model, test_dl)
                            yhat_test = torch.concat([out.cpu().unsqueeze(0) if out.dim() == 0 else out.cpu() for out in yhat_test]).squeeze()
                            inf_time = time.time()
                            yhat_train = sup_trainer.predict(ssl_model, train_dl)
                            yhat_train = torch.concat([out.cpu().unsqueeze(0) if out.dim() == 0 else out.cpu() for out in yhat_train]).squeeze()
                            print('Inference time per batch is: ', (time.time()-inf_time)*s3l_hyp_pred_head['batch_size']/len(yhat_train))
                            
                            # Clip predictions to the range of the training data
                            print('NOTE: Clipping the predictions to be within the range of the ground-truth values')
                            yhat_train = np.clip(yhat_train, y_train.min(), y_train.max())
                            yhat_test = np.clip(yhat_test, y_train.min(), y_train.max())
                                            
                        else: #'clas'
                            yhat_train_a = sup_trainer.predict(ssl_model, train_dl)
                            yhat_test_a = sup_trainer.predict(ssl_model, test_dl)
                            yhat_train_proba = F.softmax(torch.concat([out.cpu().unsqueeze(0) if out.dim() == 0 else out.cpu() for out in yhat_train_a]).squeeze(),dim=1)
                            yhat_test_proba = F.softmax(torch.concat([out.cpu().unsqueeze(0) if out.dim() == 0 else out.cpu() for out in yhat_test_a]).squeeze(),dim=1)
                            yhat_train = yhat_train_proba.argmax(1)
                            yhat_test = yhat_test_proba.argmax(1)

                # =================================================
                # DONT LOAD SSL MODEL, DIRECTLY USE LABELED DATA 
                # =================================================
                else:
                        
                    #==========================
                    # INITIALIZE
                    #==========================
                    assert (sup_model_type in ['xgb', 'mlp'],
                            f"Invalid sup_model_type: {sup_model_type}.")
                    
                    val_scaler = create_format_specific_scaler(X_train, categorical_cols, EXP_PARAM['scaler'])
                    # save the scaler model to a file
                    with open(suptrain_models_folder + scaler_to_save_name + '.pkl', 'wb') as f:
                        pickle.dump(val_scaler, f)  
                    
                    if sup_model_type == 'mlp':
                        sup_hyper_params=hypp_sup_mlp
                        # I am setting batch size based on the number of samples in the train set 
                        if EXP_PARAM['label_no'] <= 5000:
                            sup_hyper_params['batch_size'] = 10
                        else: # when it is greater
                            sup_hyper_params['batch_size'] = 50
                        print('Setting batch_size to ', sup_hyper_params['batch_size'])
                        sup_hyper_params['fc_layers'] = sup_mlp_layers
                    elif sup_model_type == 'xgb':
                        if EXP_PARAM['label_no'] <= 5000:
                            sup_hyper_params=hypp_sup_xgb_small  
                            print(sup_hyper_params)
                        else: # when it is greater
                            sup_hyper_params=hypp_sup_xgb_large
                            print(sup_hyper_params)
                                          
                    X_train = val_scaler.transform(X_train).copy()
                    X_test = val_scaler.transform(X_test).copy()
                    X_val = val_scaler.transform(X_val).copy()

                    if first_pass:
                        print('HYPERPARAMETERS: ', sup_hyper_params)
                    
                    #============================
                    # TRAIN AND MAKE PREDICTIONS
                    #============================
                    
                    model = train_model(X_train, X_val, 
                                                y_train, y_val, 
                                                sup_model_type, learning_task_type,
                                                suptrain_models_folder + suptrain_model_to_save_name,
                                                sup_hyper_params)
                    
                    if learning_task_type == 'reg':   

                        if sup_model_type == 'xgb':
                            yhat_train = model.predict(X_train).flatten()
                            yhat_test = model.predict(X_test).flatten()
                        
                        else: # pytorch mlp
                            # Set to inference mode. This disables regularization. 
                            model.eval()
                            with torch.no_grad():
                                yhat_train = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
                                yhat_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
                        
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
                
                #===============================================================
                # COMPUTE ERROR AFTER TRAINING WITH LABELED DATA  
                #===============================================================
                plot_save_path='plots/q-q_'+pt_type+'_T'+str(idx+1)+'_rand'+str(rs)+'_'+str(label_no)+'.pdf'
                train_results, test_results = compute_plot_save_error(y_train, yhat_train, 
                                                                      y_test, yhat_test, 
                                                                      learning_task, learning_task_type, 
                                                                      rs, 
                                                                      train_results, test_results, plot_save_path)
                print('\n===============================  DONE with this learning task  ===================================================')
                # end of for each learning task 
            
            print('Time to run experiments for rand itr ', rs, ' for num labeled samples ', label_no, ': ', time.time()-rand_itr_runtime)
            print('\n===============================  DONE with this random iteration  ===================================================')
            # end of for each ran itr
        
        print('============================= Train set mean over runs ===================================')
        mean_train_results = train_results.applymap(mean_array)
        print(mean_train_results)
        print('=============================== Test set results =========================================')
        print(test_results.applymap(format_array))
        print('============================= Test set mean over runs ====================================')
        mean_test_results = test_results.applymap(mean_array)
        print(test_results.applymap(mean_array))
        print('==========================================================================================')
        # Append to file
        mean_train_results.round(3).to_csv(train_results_filepath, mode='a', header=False)    
        mean_test_results.round(3).to_csv(test_results_filepath, mode='a', header=False)
        print('Time to run experiments for ', label_no, ' labeled samples: ', time.time()-num_samples_runtime)
        print('\n===============================  DONE with this num labeled samples size  ===================================================')
        # end of for each num of samples 
    
    print('Time to run the whole script: ', time.time()-script_runtime)
    print('===============================  EVERYTHING DONE  ===========================================')
    # end of function 


# call it directly
if __name__ == '__main__':
    second_phase_training(sys.argv[1])