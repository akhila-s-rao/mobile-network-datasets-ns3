# Hyperparameters for SSL using DAE 

# Hyperparameters for supervised XGB training 
hypp_sup_xgb_small={
    'learning_rate': 0.1,
    'n_estimators': 50,
    'max_depth': 3,
    'loss':{'reg':'mse',
            'clas':'categorical_crossentropy'}
} 
hypp_sup_xgb_large={
    'learning_rate': 0.01,
    'n_estimators': 200,
    'max_depth': 6,
    'loss':{'reg':'mse',
            'clas':'categorical_crossentropy'}
} 

# Hyperparameters for supervised MLP training
hypp_sup_mlp={
    'fc_layers': None, # Should be taken as input parameter to this function
    'batch_size': None, # will get set based on number of labelled samples   
    'max_epochs': 100,# 100 
    'patience': 15,
    'learning_rate': 0.0001,
    'use_batchnorm': False,
    'use_dropout': False,
    'dropout_rate': 0.0,
    'loss':{'reg':'MSELoss',#'MSELoss', 'L1Loss', SmoothL1Loss
            'clas':'categorical_crossentropy'},
    'metrics':{
               'reg':['MSELoss'],
               'clas':['Recall']},
    'out_activation':{'reg':'linear', 
                      'clas':'softmax'} 
}

s3l_hyp_pred_head={
    # I am setting this in code directly since I want to use different batch sizes for different sample siezes
    'batch_size': None,
    'max_epochs': 1, # 30
    'patience': 5,
    'loss':{'reg': 'MSELoss', #'L1Loss',
            'clas':'CrossEntropyLoss'},
    'metrics':{'reg':['r2_score'],
               'clas':['Recall']}#'F1Score' does not work needs a different dimension 
}

s3l_hyp_ssl_dae={
    'loss_fn': "MSELoss",
    #'metric': ["r2_score", "mean_absolute_error"], #
    'metric': "r2_score", #
    'hidden_dim': 200, #
    'max_epochs': 100, #
    'batch_size': 128, #
    # not used in the config yet
    'optim_hparams': {'lr': 0.0001, 'weight_decay': 0.00005},

    
    'encoder_depth': 4, #  
    'head_depth': 2, #
    'dropout_rate': 0.1, #
    
    'noise_type': "Swap", #
    'noise_ratio': 0.3 #
}

s3l_hyp_ssl_scarf={
    'loss_fn': "MSELoss",
    #'metric': ["r2_score", "mean_absolute_error"], #
    'metric': "r2_score", #
    'hidden_dim': 200,
    'max_epochs': 100,
    'batch_size': 128,
    # not used in the config yet
    'optim_hparams': {'lr': 0.0001, 'weight_decay': 0.00005},
    
    'encoder_depth': 4,
    'head_depth': 2,
    'dropout_rate': 0.1, #
    
    'corruption_rate': 0.3
}

s3l_hyp_ssl_vime={
    'loss_fn': "MSELoss",
    #'metric': ["r2_score", "mean_absolute_error"], #
    'metric': "r2_score", #
    'hidden_dim': 200, #
    'max_epochs': 100, #
    'batch_size': 128, #
    # not used in the config yet
    'optim_hparams': {'lr': 0.0001, 'weight_decay': 0.00005},

    'encoder_depth': 4, #  
    'head_depth': 2,
    'dropout_rate': 0.1, #
    
    'p_m': 0.3, # Corruption probability for self-supervised learning
    
    'alpha1': 2.0, # Hyper-parameter to control the weights of feature and mask losses
    'alpha2': 2.0, # Hyper-parameter to control the weights of feature and mask losses
    'K': 3, # Number of augmented samples
    'beta': 0.2 # Hyperparameter to control supervised and unsupervised losses
}

s3l_hyp_ssl_subtab={
    'loss_fn': "MSELoss",
    #'metric': ["r2_score", "mean_absolute_error"], #
    'metric': "r2_score", #
    'hidden_dim': 200,
    'max_epochs': 100,
    'batch_size': 128,
    # not used in the config yet
    'optim_hparams': {'lr': 0.0001, 'weight_decay': 0.00005},
    
    'encoder_depth': 2,
    'head_depth': 2,
    # NO DROPOUT

    'noise_type': "Swap",
    'mask_ratio': 0.3,
    
    'tau': 1.0,
    'use_cosine_similarity': True,
    'use_contrastive': True,
    'use_distance': True,
    'n_subsets': 4,
    'overlap_ratio': 0.75
}

s3l_hyp_ssl_switchtab={
    'loss_fn': "MSELoss",
    #'metric': ["r2_score", "mean_absolute_error"], #
    'metric': "r2_score", #
    'hidden_dim': 200,
    'max_epochs': 100,
    'batch_size': 128,
    # not used in the config yet
    'optim_hparams': {'lr': 0.0001, 'weight_decay': 0.00005},
    
    'encoder_depth': 2, # here the number does not indicate num of layers but number fo blocks 
    # NO HEAD DEPTH
    'corruption_rate': 0.3,
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
