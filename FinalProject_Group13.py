import tensorflow as tf

IMAGE_SIZE = (224, 340)
CROP_SIZE = (224, 224)
BATCH_SIZE = 128

def dataset(ds_path, *,
            train,
            preprocess_fn=None,
            image_size=IMAGE_SIZE,
            crop_size=CROP_SIZE,
            batch_size=BATCH_SIZE):
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
    
    crop_layer = tf.keras.layers.experimental.preprocessing.RandomCrop(*crop_size)
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


def model_probas(test_dataset, model):
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


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, BatchNormalization
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense, Flatten, Activation


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

    out =feature_ex_model(p1_tensor)
    feature_ex_model = Model(inputs=image_input, outputs=out)

    def naive_inception_module(layer_in, f1, f2, f3):
        # 1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
        # 3x3 conv
        conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
        # 5x5 conv
        conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
        # concatenate filters, assumes filters/channels last
        layer_out = Concatenate()([conv1, conv3, conv5, pool])
        return layer_out

    out = naive_inception_module(feature_ex_model.output, 64, 128, 32)
    num_classes = 2

    bn1 = BatchNormalization(name='BN')(out)
    f = Flatten()(bn1)
    dropout = Dropout(0.4, name='Dropout')(f)
    desne = Dense(num_classes, activation='softmax', name='Predictions')(dropout)

    model = Model(inputs=feature_ex_model.input, outputs=desne)
    return model

#AFTER HERE IS PYTORCH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

# Learning rate schedule
def scheduler(epoch, lr):
    if epoch%4 == 0 and epoch!= 0:
        lr = lr/2
    return lr   

def train_validate(model, trainloader, valloader, epochs, batch_size,
                   learning_rate, best_model_path, device, verbose):

    # Define your callbacks (save your model, patience, etc.)
    model_name = "group_13_best_model.h5"
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)

    monitor = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss',\
                                                verbose=0, save_best_only=True,\
                                                save_weights_only=False,\
                                                mode='min')


    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    model.fit(train_ds, epochs = 5, \
          verbose = 1, callbacks= [early_stop, monitor, lr_schedule], validation_data=(val_flow))


    best_loss = 1e+20
    for epoch in range(epochs):  # loop over the dataset multiple times

        # Loss function and optimizer
        criterion = model.CrossEntropyLoss()  # Loss function
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        # Training Loop
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end=' ')
        scheduler.step()

        val_loss = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
            print(f'val loss: {val_loss / i:.3f}')

            # Save best model
            if val_loss < best_loss:
                print("Saving model")
                torch.save(net.state_dict(), best_model_path)
                best_loss = val_loss

    print('Finished Training')


def test(net, testloader):

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the test images: {100 * correct / total} %')
