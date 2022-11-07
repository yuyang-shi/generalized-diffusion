import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import tensorflow_datasets as tfds

from score_sde.utils import register_dataset

import jax.numpy as jnp


class SYMSOLDataset:
    def __init__(self, shapes, batch_dims, downsample_continuous_gt=True, split='train', test_repeat=True, **kwargs):
        assert len(batch_dims) == 1
        self.batch_dims = batch_dims
        self.SHAPE_NAMES = ['tet', 'cube', 'icosa', 'cone', 'cyl', 'tetX', 'cylO', 'sphereX']
        self.split = split
        self.downsample_continuous_gt = downsample_continuous_gt

        if split == 'train':
            dataset = tfds.load('symmetric_solids', split='train[:90%]', data_dir="/data/stat-stochgenerativemodel/orie3571/tensorflow_datasets/")
        elif split == 'val':
            dataset = tfds.load('symmetric_solids', split='train[90%:]', data_dir="/data/stat-stochgenerativemodel/orie3571/tensorflow_datasets/")
        elif split == 'test':
            dataset = tfds.load('symmetric_solids', split='test', data_dir="/data/stat-stochgenerativemodel/orie3571/tensorflow_datasets/")
        else:
            raise ValueError
        
        if 'symsol1' in shapes:
            shapes = self.SHAPE_NAMES[:5]
        # if split == 'test':
        #     shapes = shapes[:1]
        shape_inds = [self.SHAPE_NAMES.index(shape) for shape in shapes]

        dataset = dataset.filter(lambda x: tf.reduce_any(tf.equal(x['label_shape'], shape_inds)))
        annotation_key = 'rotation' if split == 'train' else 'rotations_equivalent'
        
        dataset = dataset.map(
            lambda example: (example[annotation_key], tf.image.convert_image_dtype(example['image'], tf.float32)),  # tf.transpose(, (2, 0, 1))
            num_parallel_calls=tf.data.AUTOTUNE)

        print("Batch size:", self.batch_dims[0])

        if split == 'train':
            dataset = dataset.repeat().shuffle(1000).batch(self.batch_dims[0]).prefetch(tf.data.AUTOTUNE)
        else:
            if test_repeat:
                dataset = dataset.repeat().shuffle(1000).prefetch(tf.data.AUTOTUNE)
            else:
                dataset = dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE)
        
        self.dataset = tfds.as_numpy(dataset)
        self.dataset_iterator = iter(self.dataset)

    def __next__(self):
        rots, im = next(self.dataset_iterator)
        if self.split == "train":
            return rots, im
        else:
            if self.downsample_continuous_gt and rots.shape[0] > 60:
                if rots.shape[0] == 360:
                    downsample = rots.shape[0] // 60
                    rots = rots[::downsample]
                elif rots.shape[0] == 720:
                    downsample = rots.shape[0] // 30
                    rots = jnp.concatenate([rots[::downsample], rots[1::downsample]])
                return rots, jnp.broadcast_to(im, [rots.shape[0], *im.shape])
            else:
                return rots, jnp.broadcast_to(im, [rots.shape[0], *im.shape])
