from typing import Tuple

import tensorflow as tf


def dataset_split(
    dataset: tf.data.Dataset, split_fraction: float, fold: int = 0
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits the dataset into one chunk with split_fraction many elements of
    the original dataset and another chunk with size (1 - split_fraction)
    elements.

    Args:
        dataset (tf.data.Dataset): Dataset to be splitted.
        split_fraction (float): Fraction of the dataset split.
        fold (int): Which fold of the dataset, the validation set should be.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Splitted datasets tuple.
    """
    split_size = int(len(dataset) * split_fraction)
    offset_idx = fold * split_size
    val_dataset = dataset.skip(offset_idx).take(split_size)
    first_train_folds = dataset.take(offset_idx)
    last_train_folds = dataset.skip(offset_idx + split_size)
    train_dataset = first_train_folds.concatenate(last_train_folds)
    return train_dataset, val_dataset


def dataset_join(
    dataset_left: tf.data.Dataset, dataset_right: tf.data.Dataset
) -> tf.data.Dataset:
    dataset_joined = dataset_left.concatenate(dataset_right)
    return dataset_joined
