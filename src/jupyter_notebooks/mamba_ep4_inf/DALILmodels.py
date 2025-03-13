from config import CFG
from models import LMmamba, DemoModel

import tensorflow as tf

from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy



class DALILmamba(LMmamba):
    def __init__(self, batch, input_dim, input_dim_embedding, hidden_dim, num_layers, dropout, out_dim, learning_rate, weight_decay):
        super().__init__(batch, input_dim, input_dim_embedding, hidden_dim, num_layers, dropout, out_dim, learning_rate, weight_decay)

    def setup(self,stage=None):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        train_pipe = belka_pipeline(
            batch_size=CFG.BATCH_SIZE,
            num_threads=4,
            device_id=device_id,
            device='cuda',
            shard_id=shard_id,
            num_shards=num_shards,
            paths=CFG.TRAINS,
            idxs=CFG.TRAIN_IDX,
            seed=CFG.SEED + 2 + device_id*2
        )
        valid_pipe = belka_pipeline(
            batch_size=CFG.BATCH_SIZE,
            num_threads=4,
            device_id=device_id,
            device='cuda',
            shard_id=shard_id,
            num_shards=num_shards,
            paths=CFG.VALIDS,
            idxs=CFG.VALID_IDX,
            seed=CFG.SEED-2
        )

        class LightningWrapper(DALIGenericIterator):
            def __init__(self, *kargs, **kwargs):
                super().__init__(*kargs, **kwargs)
            def __next__(self):
                out = super().__next__()
                out = out[0]
                return [out[k] for k in self.output_map]


        self.train_loader = LightningWrapper(train_pipe, ['X', 'y'],reader_name='Reader', last_batch_policy=LastBatchPolicy.DROP)
        self.valid_loader = LightningWrapper(valid_pipe, ['X', 'y'],reader_name='Reader', last_batch_policy=LastBatchPolicy.PARTIAL)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.train_loader.reset()
    
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


class DALILDemoModel(DemoModel):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    def setup(self,stage=None):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        train_pipe = belka_pipeline(
            batch_size=CFG.BATCH_SIZE,
            num_threads=4,
            device_id=device_id,
            device='cuda',
            shard_id=shard_id,
            num_shards=num_shards,
            paths=CFG.TRAINS,
            idxs=CFG.TRAIN_IDX,
            seed=CFG.SEED + 2 + device_id*2
        )
        valid_pipe = belka_pipeline(
            batch_size=CFG.BATCH_SIZE,
            num_threads=4,
            device_id=device_id,
            device='cuda',
            shard_id=shard_id,
            num_shards=num_shards,
            paths=CFG.VALIDS,
            idxs=CFG.VALID_IDX,
            seed=1996
        )

        class LightningWrapper(DALIGenericIterator):
            def __init__(self, *kargs, **kwargs):
                super().__init__(*kargs, **kwargs)
            def __next__(self):
                out = super().__next__()
                out = out[0]
                return [out[k] for k in self.output_map]


        self.train_loader = LightningWrapper(train_pipe, ['X', 'y'],reader_name='Reader', last_batch_policy=LastBatchPolicy.DROP)
        self.valid_loader = LightningWrapper(valid_pipe, ['X', 'y'],reader_name='Reader', last_batch_policy=LastBatchPolicy.PARTIAL)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader




@pipeline_def
def belka_pipeline(device, paths, idxs, seed,shard_id=0, num_shards=1, is_train=True):
    device_id = Pipeline.current().device_id

    inputs = fn.readers.tfrecord(
        path = paths,
        index_path = idxs,
        features={
            "x": tfrec.FixedLenFeature([CFG.SEQ_LENGTH], tfrec.int64, 0),
            "y": tfrec.FixedLenFeature([CFG.NUM_CLASSES], tfrec.float32, .0)
        },
        random_shuffle=is_train,
        num_shards=num_shards,
        shard_id=shard_id,
        initial_fill=CFG.BATCH_SIZE,
        seed=seed,
        name='Reader'
    )
    x = inputs['x']
    y = inputs['y']
    if device=='cuda':
        x = x.gpu()
        y = y.gpu()
    return x,y
