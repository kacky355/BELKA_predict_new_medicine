from config import CFG
from DALILmodels import DALILmamba, belka_pipeline

from logger.mylogger import get_my_logger

import os
import numpy as np
import pandas as pd
import glob
import random
import time

import torch
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor,TQDMProgressBar
from torchmetrics.classification import MultilabelAveragePrecision as mAP

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_logger(name):
    now = time.localtime()
    now = time.strftime("%Y-%m-%d-%H-%M-%S", now)
    log_name = f'{name}-{now}.log'
    logger = get_my_logger(CFG.BASE_DIR, log_name)
    return logger



def calc_validation_APS(model, model_name, mode, logger):
    logger.info('validation_APS start calucurate')
    valid_pipe = belka_pipeline(
            batch_size=CFG.BATCH_SIZE,
            num_threads=4,
            device_id=0,
            device='cuda',
            paths=CFG.VALIDS,
            idxs=CFG.VALID_IDX,
            seed=CFG.SEED-2,
            is_train=False
        )

    class LightningWrapper(DALIGenericIterator):
        def __init__(self, *kargs, **kwargs):
            super().__init__(*kargs, **kwargs)
        def __next__(self):
            out = super().__next__()
            out = out[0]
            return [out[k] for k in self.output_map]

    valid_loader = LightningWrapper(valid_pipe, ['X', 'y'],reader_name='Reader', last_batch_policy=LastBatchPolicy.PARTIAL)

    metrics_ap = mAP(3,thresholds=None, average='none')
    metrics_map = mAP(3, thresholds=None, average='micro')
    all_preds = []
    all_y = []
    model.to(0)
    model.eval()
    with torch.no_grad():
        for X, y in valid_loader:
            oof = model(X)
            all_preds.append(oof)
            all_y.append(y)
    preds = torch.cat(all_preds, 0)
    y_eval = torch.cat(all_y, 0).to(torch.int)
    
    APs = metrics_ap(preds, y_eval)
    for i in range(3):
        logger.info(f'val_AP_bind{i} : {APs[i]}')
    meanAP = metrics_map(preds, y_eval)
    logger.info(f'valid_results: CV score = {meanAP}')
    preds = preds.clone().to('cpu').detach().numpy()
    val_results = pd.DataFrame({f'bind{i}': preds[:, i] for i in range(3)})
    val_results.to_csv(os.path.join(CFG.BASE_DIR, f'val_results_{model_name}_{mode}.csv'))
    logger.info(f'val_results write complite!\nfile_name: val_results_{model_name}_{mode}.csv')

def make_submit(model, model_name, mode, logger):
    logger.info('load test data')
    test_data = pd.read_csv('/kaggle/input/leash-BELKA/test.csv')
    test_data = TensorDataset(torch.tensor(test_data.values, dtype=torch.int))
    test_loader= DataLoader(test_data, batch_size=CFG.BATCH_SIZE)
    logger.info('predict test data')
    test_preds=[]
    model.to(0)
    model.eval()
    with torch.no_grad():
        for (X,) in test_loader:
            X = X.to(0)
            oof = model(X)
            test_preds.append(oof)
    preds= torch.cat(test_preds, 0).detach().numpy()

    logger.info('writing submission.csv start')
    tst = pd.read_parquet('/kaggle/input/leash-BELKA/test.parquet')
    tst['binds'] = 0
    tst.loc[tst['protein_name']=='BRD4', 'binds'] = preds[(tst['protein_name']=='BRD4').values, 0]
    tst.loc[tst['protein_name']=='HSA', 'binds'] = preds[(tst['protein_name']=='HSA').values, 1]
    tst.loc[tst['protein_name']=='sEH', 'binds'] = preds[(tst['protein_name']=='sEH').values, 2]
    tst[['id', 'binds']].to_csv(f'submission_{model_name}_{mode}.csv', index = False)
    logger.info('writing complete')



if __name__ == '__main__':
    logger = set_logger(CFG.MODEL_NAME)

    set_seeds(seed= CFG.SEED)
    logger.info(f'set seed: {CFG.SEED}')

    model_module = DALILmamba(**CFG.MODEL_PARAM)
    logger.info(f'model has made.\n model:{model_module}')


    early_stop_callback = EarlyStopping(
        monitor= 'val_loss',
        mode= 'min',
        patience= 3,
        verbose= True
    )

#     loss_checkpoint_callback = ModelCheckpoint(
#         dirpath= f'{CFG.BASE_DIR}/models/',
#         filename= f'model-{CFG.MODEL_NAME}-{{epoch}}-{{val_loss:2f}}',
#         monitor= 'val_loss',
#         mode='min'
#         save_top_k= 1,
#         verbose= True,
#     )
    mAP_checkpoint_callback = ModelCheckpoint(
        dirpath= f'{CFG.BASE_DIR}/models/',
        filename= f'model-{CFG.MODEL_NAME}-{{epoch}}-{{val_mAP:2f}}',
        monitor= 'val_mAP',
        mode='max',
        save_top_k= 1,
        verbose= True,
    )

    progress_bar_callback = TQDMProgressBar(refresh_rate=1)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [
        early_stop_callback,
#         loss_checkpoint_callback,
        mAP_checkpoint_callback,
        progress_bar_callback,
        lr_monitor,
    ]
    logger.info('set callbacks')
    
    ckpts = glob.glob(os.path.join('/kaggle/input/mamba-ep0/models', '*.ckpt'))
    ckpts.sort()
    latest_checkpoint = ckpts[-1] if len(ckpts) != 0 else None

    trainer = L.Trainer(
            max_epochs= CFG.EPOCHS,
#             max_time='00:08:30:00',
            callbacks= callbacks,
            accelerator= 'auto',
            enable_progress_bar= True,
            devices= 'auto',
            strategy='ddp',
        )
    logger.info('trainer has made')   
    if latest_checkpoint:
        logger.info(f'load ckpt from {latest_checkpoint}')
    logger.info('training begin')
    trainer.fit(model_module, ckpt_path=latest_checkpoint)
    logger.info('training finish!')
#     model_module = DALILmamba.load_from_checkpoint(loss_checkpoint_callback.best_model_path)
#     calc_validation_APS(model_module, CFG.MODEL_NAME, 'loss', logger)
#     make_submit(model_module, CFG.MODEL_NAME, 'loss', logger)
    
    model_module = DALILmamba.load_from_checkpoint(mAP_checkpoint_callback.best_model_path)
    calc_validation_APS(model_module, CFG.MODEL_NAME, 'mAP', logger)
    make_submit(model_module, CFG.MODEL_NAME, 'mAP', logger)
