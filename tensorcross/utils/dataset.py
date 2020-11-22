from typing import Tuple

import tensorflow as tf


def dataset_split(
    dataset: tf.data.Dataset,
    split_size: float
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits the dataset into one chunk with spliz_size many elements and
    another chunk with size (1 - split_size) elements.

    Args:
        dataset (tf.data.Dataset): Dataset to be splitted.
        split_size (float): Fraction of the dataset split.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Splitted datasets-
    """
    N = int(len(dataset) * split_size)
    validation_dataset = dataset.take(N)
    train_dataset = dataset.skip(N)
    return train_dataset, validation_dataset
