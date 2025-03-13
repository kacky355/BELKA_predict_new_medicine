import os
import glob

class CFG:
    DEBUG = False
    MODEL_NAME = 'mamba'

    EPOCHS = 4
    BATCH_SIZE = 1024
    NBR_FOLDS = 15
    NUM_TRAINS = 91_854_569
    NUM_VALIDS = 6_561_041
    STEPS_PER_EPOCH_TRAIN = (NUM_TRAINS -1) //BATCH_SIZE +1
    STEPS_PER_EPOCH_VALID = (NUM_VALIDS -1) //BATCH_SIZE +1


    SELECTED_FOLDS = [0]

    BASE_DIR = '/kaggle/working/'
    DATA_SOURCE = '/kaggle/input/belka-train-valid-tfrecords-1d-preprocessed'
    DATA_SOURCE_DOWNSAMPLE = '/kaggle/input/be-train-tfrec-downsample'
#     TRAINS = glob.glob('/kaggle/input/be-train-tfrec-downsample/trains/BELKA_train_*.tfrecord')
    TRAINS = glob.glob(os.path.join(DATA_SOURCE_DOWNSAMPLE, 'trains','BELKA_train_*.tfrecord'))
    TRAINS.sort()
    TRAIN_IDX = glob.glob(os.path.join(DATA_SOURCE_DOWNSAMPLE, 'tf_idx', 'train_*.idx'))
    TRAIN_IDX.sort()
    VALIDS = glob.glob(os.path.join(DATA_SOURCE, 'valid/*'))
    VALIDS.sort()
    VALID_IDX = glob.glob(os.path.join(DATA_SOURCE, 'tf_idx', 'valid_*.idx'))
    VALID_IDX.sort()

    SEED = 2024


    FEATURES = [f'enc{i}' for i in range(142)]
    TARGETS = ['bind1', 'bind2', 'bind3']
    COLUMNS = FEATURES + TARGETS

    NUM_CLASSES = 3
    SEQ_LENGTH = 142


    MODEL_PARAM = {
        'batch': BATCH_SIZE,
        'input_dim': SEQ_LENGTH,
        'hidden_dim': 256 if not DEBUG else 16,
        'input_dim_embedding': 37,
        'dropout': 0.1,
        'num_layers': 8 if not DEBUG else 1,
        'out_dim': 3,
        'learning_rate' : 1e-5,
        'weight_decay' : 0.05
    }


    if DEBUG:
        EPOCHS = 2
        TRAINS = TRAINS[:1]
        TRAIN_IDX = TRAIN_IDX[:1]
        VALIDS = VALIDS[0]
        VALID_IDX = VALID_IDX[0]
