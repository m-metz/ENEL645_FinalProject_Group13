import math
import random
import shutil
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pylab as plt

'''
Change back to tensorflow.keras to turn on lazy loading of imports and to
match the exact keras version that tensorflow uses as of tensorflow 2.10
'''
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet import preprocess_input as resnet_preprocess
from keras.models import Model
from keras.models import Model, Sequential
from keras.layers import Lambda, Input
from keras.layers import Conv2D, MaxPooling2D, Concatenate, BatchNormalization
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Dropout, GlobalAveragePooling2D, Dense, Flatten, Activation


AUTOTUNE = tf.data.AUTOTUNE


'''
---
Configuration params
---

Changing these after importing doesn't affect function defaults, but do affect
functions that use these configurations directly.
'''

dataset_path = Path(
    'BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
)
'''
The path where the two class folders (benign, malignant) of images reside.

raw strings can be used in-case you use a Windows path with `\`.

If you want any other paths in this script to be cross platform, you *must* use
the forward slash `/` to make the paths work on Linux or Mac. But if you are
just using a path on only a Windows machine (like this dataset_path) you can
use `\`.

Also note you can't end a raw string with a `\` (and don't need to in this case
as we just need the path up to the folder)
'''

class_list = ['benign', 'malignant']
'''
List of expected class subfolders in the dataset folder.
'''

train_split = 0.6
'''
Set the train split. Train, validation, test split must add up to approximately 1.0.
'''

validation_split = 0.1
'''
Set the validation split. Train, validation, test split must add up to approximately 1.0.
'''

test_split = 1.0 - validation_split - train_split
'''
Set the test split. Train, validation, test split must add up to approximately 1.0.
'''

random_seed = 154
'''
Used for configuring a consistent random_seed where we need randomness with
reproducable results, like when shuffling the order of images.
'''

batch_size = 128
'''
The batch_size for training.
'''

image_size = (224, 340)
'''
The image size of all images in the dataset.
'''

crop_size = (224, 224)
'''
The size to randomly crop all images to during preprocessing (including train, validation, test).
'''


'''
---
Functions
---
'''


def stratified_train_val_test_split_into_folders(
        dataset_path,
        *,
        class_list=class_list,
        split_data_path=None,
        move=False,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        random_seed=random_seed):
    """
    Loops through the `class_list` and splits the data set into train, test,
    and validation datasets. The images will be in `split_data_path`/`

    Args:
        dataset_path (Path, optional): The folder that contains the class folders with pngs in the class folders or any folder below. Defaults to DATASET_PATH.
        class_list (list, optional): List of expected class subfolders. Defaults to class_list.
        split_data_path (Path, optional): Where to output the split data. Defaults to None (meaning dataset_path/'split_data').
        move (bool, optional): Move files from `dataset_path` if True, else copy the files. Defaults to False.
        train_split (float, optional): Amount to split into training. Defaults to train_split.
        validation_split (float, optional): Amount to split into validation. Defaults to validation_split.
        test_split (float, optional): Amount to split into test. Defaults to test_split.
        random_seed (int, optional): random seed to use for shuffling. Defaults to random_seed.

    Raises:
        ValueError: All splits must add up to approximately 1.0, if they don't this is raised.

    Returns:
        list(str): A list of strings, one each for train, validation, test path.
    """

    TRAIN_FOLDER_NAME = 'training'
    VALIDATION_FOLDER_NAME = 'validation'
    TEST_FOLDER_NAME = 'test'

    if split_data_path is None:
        split_data_path = dataset_path / 'split_data'

    split_total = train_split + validation_split + test_split
    EXPECTED_SPLIT_TOTAL = 1.0
    if not math.isclose(split_total, EXPECTED_SPLIT_TOTAL):
        raise ValueError(
            'train_split + validation_split + test_split ({}) is not approximately = {}'.format(split_total, EXPECTED_SPLIT_TOTAL))

    copy_move_str = 'Copying'
    if move:
        copy_move_str = 'Moving'

    development_split = train_split + validation_split

    destination_paths = []

    allow_move_or_copy = True

    if split_data_path.exists():
        print(
            f"Not {copy_move_str.lower()} files as {split_data_path} already esists")
        allow_move_or_copy = False

    split_data_path.mkdir(parents=True, exist_ok=True)

    for class_index, class_ in enumerate(class_list):
        class_images = glob(
            str(dataset_path / class_ / '**/*.png'), recursive=True)

        # Shuffles the list in place.
        random.Random(random_seed).shuffle(class_images)

        development_length = int(development_split * len(class_images))

        print(f'Development {class_} set length: {development_length}')
        print(
            f'Test {class_} set length: {len(class_images) - development_length}')

        development_class_image_paths = class_images[:development_length]
        test_class_image_paths = class_images[development_length:]

        print(
            f'Development {class_} image count: {len(development_class_image_paths)}')
        print(f'Test {class_} image count: {len(test_class_image_paths)}')

        '''
        / does float division in python3 and we expect these numbers to be float
        anyways.

        TRAIN_SPLIT is relative to DEVELOPMENT_SPLIT images because we are working
        with an images subset, and the numbers are absolute to the total dataset.
        '''
        training_length = int(train_split / development_split *
                              len(development_class_image_paths))

        print(f'Training {class_} set length: {training_length}')
        print(
            f'Validation {class_} set length: {len(development_class_image_paths) - training_length}')

        training_class_image_paths = development_class_image_paths[:training_length]
        validation_class_image_paths = development_class_image_paths[training_length:]
        print(
            f'Training {class_} image count: {len(training_class_image_paths)}')
        print(
            f'Validation {class_} image count: {len(validation_class_image_paths)}')

        split_folder_name_split_image_class_paths_dict = {
            TRAIN_FOLDER_NAME: training_class_image_paths,
            VALIDATION_FOLDER_NAME: validation_class_image_paths,
            TEST_FOLDER_NAME: test_class_image_paths
        }

        print()

        for split_folder_name, split_class_image_paths in split_folder_name_split_image_class_paths_dict.items():
            split_path = split_data_path / split_folder_name
            destination_path: Path = split_path / class_

            '''
            Only append destination paths and make the split folders on the
            first class_ iteration. We don't want duplicate folders.
            '''
            if class_index == 0:
                destination_paths.append(str(split_path))
                split_path.mkdir(parents=False, exist_ok=True)

            '''
            Make the class folder in each split folder
            '''
            destination_path.mkdir(parents=False, exist_ok=True)

            if allow_move_or_copy:
                print(
                    f'{copy_move_str} {split_folder_name} files from {dataset_path} to {destination_path}')
                for split_class_image_path in split_class_image_paths:
                    if move == True:
                        shutil.move(split_class_image_path,
                                    str(destination_path))
                    else:
                        shutil.copy(split_class_image_path,
                                    str(destination_path))

        '''
        If not the last iteration and allow_move_or_copy...
        '''
        if class_index != len(class_list) and allow_move_or_copy:
            print()

    return destination_paths


def preprocess_train_val(
        training_dataset_path,
        validation_dataset_path,
        *,
        image_size=image_size,
        crop_size=crop_size,
        batch_size=batch_size
) -> tf.data.Dataset:
    """ 
    This function will take parameters for the datas file path along with the image, crop, and batch size. It will then perform the training
    set's cropping and data augmentation and return the dataset once it is transformed.
    """

    crop_layer = tf.keras.layers.RandomCrop(*crop_size, seed=random_seed)
    augmentation_layer = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation((-0.2, 0.2), seed=random_seed),
            tf.keras.layers.RandomContrast(0.1, seed=random_seed),
            tf.keras.layers.RandomHeight(0.2, seed=random_seed),
            tf.keras.layers.RandomWidth(0.2, seed=random_seed),
            tf.keras.layers.RandomZoom(0.2, seed=random_seed),
            tf.keras.layers.Resizing(224, 340, crop_to_aspect_ratio=True)


        ]
    )

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_dataset_path,
        shuffle=True,
        label_mode='categorical',
        seed=random_seed,
        batch_size=batch_size,
        image_size=image_size)

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        validation_dataset_path,
        shuffle=False,
        label_mode='categorical',
        seed=random_seed,
        batch_size=batch_size,
        image_size=image_size)

    train_ds = train_ds.map(
        lambda image, label: (
            augmentation_layer(image, training=True),
            label),
        num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.map(
        lambda image, label: (
            crop_layer(image, training=True),
            label),
        num_parallel_calls=AUTOTUNE)

    validation_ds = validation_ds.map(
        lambda image, label: (
            crop_layer(image, training=True),
            label),
        num_parallel_calls=AUTOTUNE)

    return train_ds.prefetch(buffer_size=AUTOTUNE), validation_ds.prefetch(buffer_size=AUTOTUNE)


def preprocess_test(
        path,
        *,
        image_size=image_size,
        crop_size=crop_size,
        batch_size=batch_size
) -> tf.data.Dataset:

    crop_layer = tf.keras.layers.CenterCrop(*crop_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        label_mode='categorical',
        seed=random_seed,
        batch_size=batch_size,
        image_size=image_size)

    test_ds = test_ds.map(
        lambda image, label: (
            crop_layer(image, training=True),
            label),
        num_parallel_calls=AUTOTUNE)

    return test_ds.prefetch(buffer_size=AUTOTUNE)


# Models

# Classic Resnet50 with transfer learning

def resnet50_builder():
    base_model = tf.keras.applications.resnet50.ResNet50(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False)
    base_model.trainable = False

    x1 = base_model(base_model.input, training=False)
    x2 = tf.keras.layers.Flatten()(x1)
    num_classes = 2

    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x2)
    model = tf.keras.Model(inputs=base_model.input, outputs=out)

    return model

# Vgg16 + Naive Inception Block


def vgginnet_builder():
    base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

    layer_name = 'block4_pool'
    feature_ex_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer_name).output,
        name='vgg16_features')
    feature_ex_model.trainable = False

    p1_layer = Lambda(vgg_preprocess, name='VGG_Preprocess')
    image_input = Input((224, 224, 3), name='Image_Input')
    p1_tensor = p1_layer(image_input)

    out = feature_ex_model(p1_tensor)
    feature_ex_model = Model(inputs=image_input, outputs=out)

    def naive_inception_module(layer_in, f1, f2, f3):
        # 1x1 conv
        conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
        # 3x3 conv
        conv3 = Conv2D(f2, (3, 3), padding='same', activation='relu')(layer_in)
        # 5x5 conv
        conv5 = Conv2D(f3, (5, 5), padding='same', activation='relu')(layer_in)
        # 3x3 max pooling
        pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
        # concatenate filters, assumes filters/channels last
        layer_out = Concatenate()([conv1, conv3, conv5, pool])
        return layer_out

    out = naive_inception_module(feature_ex_model.output, 64, 128, 32)
    num_classes = 2

    bn1 = BatchNormalization(name='BN')(out)
    f = Flatten()(bn1)
    dropout = Dropout(0.4, name='Dropout')(f)
    desne = Dense(
        num_classes,
        activation='softmax',
        name='Predictions')(dropout)

    model = Model(inputs=feature_ex_model.input, outputs=desne)
    return model

# Resnet50 + Naive Inception Block


def resnetnaive_builder(layer_name):
    """
    function that inserts a naive block of layers after a specific block of a pretrained resnet50's architecture. 
    Params:
    layer_name - name of resnet50's block where Naive Inception will be inserted. Remaining blocks of resnet50 are discarded. 
    Classification layer is connected right after Naive Block. 
    Possible values:
    'conv2_block3_out' - Naive inception block will be inserted after second convolutional block.
    'conv3_block4_out' - Naive inception block will be inserted after third convolutional block.
    'conv4_block6_out' - Naive inception block will be inserted after fourth convolutional block.
    'conv5_block3_out' - Naive inception block will be inserted after fifth convolutional block.

    """
    base_model = tf.keras.applications.resnet50.ResNet50(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False)

    feature_ex_model = Model(inputs=base_model.input,
                             outputs=base_model.get_layer(layer_name).output,
                             name='resnet50_features')
    feature_ex_model.trainable = False

    p1_layer = Lambda(resnet_preprocess, name='Resnet_Preprocess')
    image_input = Input((224, 224, 3), name='Image_Input')
    p1_tensor = p1_layer(image_input)

    out = feature_ex_model(p1_tensor)
    feature_ex_model = Model(inputs=image_input, outputs=out)

    def naive_inception_module(layer_in, f1, f2, f3):
        # 1x1 conv
        conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
        # 3x3 conv
        conv3 = Conv2D(f2, (3, 3), padding='same', activation='relu')(layer_in)
        # 5x5 conv
        conv5 = Conv2D(f3, (5, 5), padding='same', activation='relu')(layer_in)
        # 3x3 max pooling
        pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
        # concatenate filters, assumes filters/channels last
        layer_out = Concatenate()([conv1, conv3, conv5, pool])
        return layer_out

    out = naive_inception_module(feature_ex_model.output, 64, 128, 32)
    num_classes = 2

    bn1 = BatchNormalization(name='BN')(out)
    f = Flatten()(bn1)
    dropout = Dropout(0.4, name='Dropout')(f)
    dense = Dense(num_classes, activation='softmax',
                  name='Predictions')(dropout)

    model = Model(inputs=feature_ex_model.input, outputs=dense)
    return model


def train_validate(model: Model, train_ds, val_ds, *, best_model_file='group_13_best_model.h5', epochs=100, learning_rate=1e-3):

    #
    # Define your callbacks (save best model, early stopping, learning rate scheduler)
    #

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20)

    monitor = tf.keras.callbacks.ModelCheckpoint(
        best_model_file,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='min')

    # Learning rate schedule
    # Reduce learning rate every M epochs after N epochs.
    def scheduler(epoch, lr):
        if epoch % 4 == 0 and epoch >= 15:
            lr = lr/2
        return lr

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        scheduler,
        verbose=0)

    # Show model summary before training.
    print(model.summary())

    #
    # Configure and train the model
    #

    # Define optimizer, loss function, and metrics.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tfa.metrics.F1Score(num_classes=2, name='f1_score'),
                 'mae'])

    model.fit(
        train_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stop, monitor, lr_schedule],
        validation_data=(val_ds))


def test(model_file, test_ds: tf.data.Dataset):
    """
    Args:
        test_ds: Expects test_ds to be preprocessed for pre-trained model.
    """
    model: Model = tf.keras.models.load_model(model_file)
    metrics = model.evaluate(test_ds)

    '''
    Rename f1_score to include class names, as f1 scores per class are outputted.
    '''
    metric_names = []
    for name in model.metrics_names:
        if name.lower() == 'f1_score':
            name = '_'.join([name] + class_list)
        metric_names.append(name)

    return dict(zip(metric_names, metrics))
