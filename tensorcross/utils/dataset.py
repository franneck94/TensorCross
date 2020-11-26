from typing import Tuple

import tensorflow as tf


def dataset_split(
    dataset: tf.data.Dataset,
    split_fraction: float
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits the dataset into one chunk with val_size many elements and
    another chunk with size (1 - split_fraction) elements.

    Args:
        dataset (tf.data.Dataset): Dataset to be splitted.
        split_fraction (float): Fraction of the dataset split.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Splitted datasets-
    """
    val_size = int(len(dataset) * split_fraction)
    val_dataset = dataset.take(val_size)
    train_dataset = dataset.skip(val_size)
    return train_dataset, val_dataset
