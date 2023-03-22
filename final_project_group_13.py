import math
import random
import shutil
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

'''
Change back to tensorflow.keras to turn on lazy loading of imports and to
match the exact keras version that tensorflow uses as of tensorflow 2.10
'''
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.models import Model
from keras.layers import Lambda, Input
from keras.layers import Conv2D, MaxPooling2D, Concatenate, BatchNormalization
from keras.layers import Dropout, Dense, Flatten

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
just using a path on only a Windows machine (like this DATASET_PATH) you can
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
Used for configuring a consistent RANDOM_SEED where we need randomness with
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

model_name = "group_13_best_model.h5"
'''
The name of the file to save the best model to (in .h5 format).
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
        random_seed=random_seed
):
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
        batch_size=batch_size):
    """ 
    This function will take parameters for the datas file path along with the image, crop, and batch size. It will then perform the training
    set's cropping and data augmentation and return the dataset once it is transformed.
    """

    crop_layer = tf.keras.layers.CenterCrop(*crop_size)
    augmentation_layer = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation((-0.2, 0.2), seed=random_seed),
            tf.keras.layers.RandomContrast(0.1, seed=random_seed),
            tf.keras.layers.RandomHeight(0.2, seed=random_seed),
            tf.keras.layers.RandomWidth(0.2, seed=random_seed),
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
        batch_size=batch_size):

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


def dataset(ds_path, *,
            train,
            preprocess_fn=None,
            image_size=image_size,
            crop_size=crop_size,
            batch_size=batch_size):
    """Returns a tf.data.Dataset pipeline suitable for training / inference.
    Training pipeline: uses data augmentation, random crops.
    Inference (test, val) pipeline: uses only central crop.

    Preprocessing function is applied at the end of each pipeline.
    """

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        ds_path, shuffle=train, label_mode='categorical',
        batch_size=batch_size, image_size=image_size)

    gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        zoom_range=0.2)

    @tf.function
    def augment(images, labels):
        aug_images = tf.map_fn(lambda image: tf.numpy_function(gen.random_transform,
                                                               [image],
                                                               tf.float32),
                               images)
        aug_images = tf.ensure_shape(aug_images, images.shape)
        return aug_images, labels

    crop_layer = tf.keras.layers.experimental.preprocessing.RandomCrop(
        *crop_size)

    @tf.function
    def crop(images, labels):
        cropped_images = crop_layer(images, training=train)
        return cropped_images, labels

    if train:
        ds = ds.map(augment, tf.data.experimental.AUTOTUNE)
    ds = ds.map(crop, tf.data.experimental.AUTOTUNE)
    if preprocess_fn:
        ds = ds.map(preprocess_fn)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def model_probas(test_dataset, model: Model):
    """Returns the predicted probabilities and the true labels
    for a given (inference) dataset on a given model."""
    y_test, y_probas = [], []

    for image_batch, label_batch in test_dataset:
        y_test.append(label_batch)
        y_probas.append(model.predict(image_batch))

    y_test, y_probas = (
        tf.concat(y_test, axis=0),
        tf.concat(y_probas, axis=0))

    return {
        'y_test': y_test,
        'y_probas': y_probas
    }


def vgginnet_builder():
    base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

    layer_name = 'block4_pool'
    feature_ex_model = Model(inputs=base_model.input,
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
    desne = Dense(num_classes, activation='softmax',
                  name='Predictions')(dropout)

    model = Model(inputs=feature_ex_model.input, outputs=desne)
    return model


def get_data_loaders(images_path, val_split, test_split, batch_size=32, verbose=True):
    """
    These function generates the data loaders for our problem. It assumes paths are
    defined by "/" and image files are jpg. Each subfolder in the images_path 
    represents a different class.

    Args:
        images_path (_type_): Path to folders containing images of each class.
        val_split (_type_): percentage of data to be used in the val set
        test_split (_type_): percentage of data to be used in the val set
        verbose (_type_): debug flag

    Returns:
        DataLoader: Train, validation and test data laoders.
    """

    return trainloader, valloader, testloader


def train_validate(model: Model, train_ds, val_ds, epochs=5, learning_rate=1e-4):

    #
    # Define your callbacks (save best model, early stopping, learning rate scheduler)
    #

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20)

    monitor = tf.keras.callbacks.ModelCheckpoint(
        model_name, monitor='val_loss',
        verbose=0, save_best_only=True,
        save_weights_only=False,
        mode='min')

    # Learning rate schedule
    # Reduce learning rate every 4 epochs.
    def scheduler(epoch, lr):
        if epoch % 4 == 0 and epoch != 0:
            lr = lr/2
        return lr
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        scheduler, verbose=0)

    # Show model summary before training.
    print(model.summary())

    #
    # Configure and train the model
    #

    # Define optimizer, loss function, and metrics.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, epochs=epochs,
              verbose=1, callbacks=[early_stop, monitor, lr_schedule], validation_data=(val_ds))


def test(model: Model, test_ds: tf.data.Dataset):
    """
    Args:
        test_ds: Expects test_ds to be preprocessed for pre-trained model.
    """

    model.load_weights(model_name)
    metrics = model.evaluate(test_ds)

    Ypred = model.predict(test_ds).argmax(axis=1)
    label_batch_list = []
    for _, label_batch in test_ds:
        label_batch_list.append(label_batch)
    Y_test_t = tf.concat(label_batch_list, axis=0)
    Y_test = Y_test_t.numpy()

    wrong_indexes = np.where(Ypred != Y_test)[0]

    return metrics, wrong_indexes
