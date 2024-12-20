import tensorflow as tf

import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import os

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 shard_id, num_shards, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size,
                                              num_threads,
                                              device_id,
                                              seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir+"/train",
                                    shard_id=shard_id,
                                    num_shards=num_shards,
                                    random_shuffle=True,
                                    pad_last_batch=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.RandomResizedCrop(device=dali_device, size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_layout=types.NHWC,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, shard_id, num_shards):
        super(HybridValPipe, self).__init__(batch_size,
                                           num_threads,
                                            device_id,
                                            seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir+"/val",
                                    shard_id=shard_id,
                                    num_shards=num_shards,
                                    random_shuffle=False,
                                    pad_last_batch=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu",
                              resize_shorter=crop,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_layout=types.NHWC,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

def dataset_options():
    options = tf.data.Options()
    try:
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.autotune = False   
    except:
        print('Could not set TF Dataset Options')

    return options

def get_train_val(data_path, batch_size, num_threads=4, device_id=0, crop=224, shard_id=0, num_shards=1):

    # Parameters for DALI TF DATASET
    shapes = (
        (batch_size, crop, crop, 3),
        (batch_size, 1)
        )
    dtypes = (
        tf.float32,
        tf.int32
        )

    imagenet_pipeline = HybridTrainPipe(batch_size, num_threads, device_id, data_path, crop, shard_id, num_shards, False)

    train_dataset = dali_tf.DALIDataset(
        pipeline=imagenet_pipeline,
        batch_size=batch_size,
        output_shapes=shapes,
        output_dtypes=dtypes,
        num_threads=num_threads,
        device_id=device_id)

    train_dataset = train_dataset.with_options(dataset_options())

    imagenet_val_pipeline = HybridValPipe(batch_size, num_threads, device_id, data_path, crop, shard_id, num_shards)

    val_dataset = dali_tf.DALIDataset(
        pipeline=imagenet_val_pipeline,
        batch_size=batch_size,
        output_shapes=shapes,
        output_dtypes=dtypes,
        num_threads=num_threads,
        device_id=device_id)

    val_dataset = val_dataset.with_options(dataset_options())

    return train_dataset, val_dataset
