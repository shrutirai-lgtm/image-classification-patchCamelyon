import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Tuple

BATCH_SIZE   = 256          # 128–512 fits well on most Apple / CUDA GPUs
SHUFFLE_BUF  = 50_000       # ≫ 1 batch avoids class skew
AUTOTUNE     = tf.data.AUTOTUNE
CACHE_IN_RAM = True         # Set False if RAM is limited

class PatchCamelyonDataLoader:
    def __init__(self, dataset_name: str = "patch_camelyon", seed: int = 42):
        self.dataset_name = dataset_name
        self.seed = seed
        self._load_data()

    @staticmethod
    def _prep(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def _configure(self, ds: tf.data.Dataset, *, training: bool = False
                   ) -> tf.data.Dataset:
        if training:
            ds = ds.shuffle(SHUFFLE_BUF, seed = self.seed)

        if CACHE_IN_RAM:
            ds = ds.cache()

        ds = (
            ds.map(self._prep, num_parallel_calls=AUTOTUNE)
              .batch(BATCH_SIZE, drop_remainder=training)
              .prefetch(AUTOTUNE)
        )
        return ds

    def _load_data(self):
        (self.ds_train, self.ds_val, self.ds_test), self.ds_info = tfds.load(
            self.dataset_name,
            split=["train", "validation", "test"],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            download=True
        )
        BATCH_SIZE = 32
        AUTOTUNE = tf.data.AUTOTUNE

        self.ds_train = self.ds_train.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)
        self.ds_val = self.ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        self.ds_test = self.ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    def get_train_data(self):
        return self.ds_train

    def get_val_data(self):
        return self.ds_val

    def get_test_data(self):
        return self.ds_test
    
    def get_info(self):
        return self.ds_info
    
    def get_image_shape(self):
        return self.ds_info.features['image'].shape
    
    def get_num_classes(self):
        return self.ds_info.features['label'].num_classes

if __name__ == "__main__":
    data_loader = PatchCamelyonDataLoader()
    print(data_loader.get_info())