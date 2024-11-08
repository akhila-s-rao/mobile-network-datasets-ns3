import warnings
import logging
import os

# DEBUG MODE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
    
from s3l_training import s3l_training

s3l_training(iteration='FPitr3',
             pt_type='subtab',
             pretrain=True, # if True First Phase training
             use_pretrained_model=False,
             freeze_encoder=False)


