# Assignment4
Assignment4-ResNet on CIFAR10
  1. Understood the ResNet architecture 
  2. Ran with Baseline code and acheived accuracy up to 82%
  3. Tuned the Network by changing learning rate . Varied learning rate as per loss instead of keeping leraning rate same for fixed number of epochs.  This increased the accuracy up to 88% 
  4.  Implemented cut out logic as part of Data augmentation/Regularization technique.
      This increased the accuracy to 89.5%   
========================================================================================================================================
  
  from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 50
data_augmentation = True
num_classes = 10
random_erasing = True
pixel_level = False


# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
 #   lr = 1e-3
 #   if epoch > 180:
 #       lr *= 0.5e-3
 #   elif epoch > 160:
 #       lr *= 1e-3
 #   elif epoch > 120:
 #       lr *= 1e-2
 #   elif epoch > 80:
 #       lr *= 1e-1
    return round(0.003 * 1/(1 + 0.319 * epoch), 10)
    print('Learning rate: ', lr)
    return lr

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.004),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        #preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0,
        # set function that will be applied on each input
        #preprocessing_function=None,
        preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=pixel_level))

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
Using TensorFlow backend.
The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.
We recommend you upgrade now or ensure your notebook will continue to use TensorFlow 1.x via the %tensorflow_version 1.x magic: more info.

Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170500096/170498071 [==============================] - 4s 0us/step
x_train shape: (50000, 32, 32, 3)
50000 train samples
10000 test samples
y_train shape: (50000, 1)
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:66: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:541: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4479: The name tf.truncated_normal is deprecated. Please use tf.random.truncated_normal instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:197: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:203: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:2041: The name tf.nn.fused_batch_norm is deprecated. Please use tf.compat.v1.nn.fused_batch_norm instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:148: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4271: The name tf.nn.avg_pool is deprecated. Please use tf.nn.avg_pool2d instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/optimizers.py:793: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3576: The name tf.log is deprecated. Please use tf.math.log instead.

Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 16)   448         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 16)   64          conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 16)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 16)   2320        activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 16)   64          conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 16)   0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 16)   2320        activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 32, 16)   64          conv2d_3[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 32, 16)   0           activation_1[0][0]               
                                                                 batch_normalization_3[0][0]      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 16)   0           add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 16)   2320        activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 32, 16)   64          conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 32, 16)   0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 16)   2320        activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 16)   64          conv2d_5[0][0]                   
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 32, 16)   0           activation_3[0][0]               
                                                                 batch_normalization_5[0][0]      
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 16)   0           add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 16)   2320        activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 16)   64          conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 16)   0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 16)   2320        activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 16)   64          conv2d_7[0][0]                   
__________________________________________________________________________________________________
add_3 (Add)                     (None, 32, 32, 16)   0           activation_5[0][0]               
                                                                 batch_normalization_7[0][0]      
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 16)   0           add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 16, 16, 32)   4640        activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 16, 16, 32)   128         conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 16, 16, 32)   0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 16, 32)   9248        activation_8[0][0]               
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 16, 32)   544         activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 16, 16, 32)   128         conv2d_9[0][0]                   
__________________________________________________________________________________________________
add_4 (Add)                     (None, 16, 16, 32)   0           conv2d_10[0][0]                  
                                                                 batch_normalization_9[0][0]      
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 16, 16, 32)   0           add_4[0][0]                      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 16, 32)   9248        activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 16, 16, 32)   128         conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 16, 16, 32)   0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 16, 32)   9248        activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 16, 16, 32)   128         conv2d_12[0][0]                  
__________________________________________________________________________________________________
add_5 (Add)                     (None, 16, 16, 32)   0           activation_9[0][0]               
                                                                 batch_normalization_11[0][0]     
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 16, 16, 32)   0           add_5[0][0]                      
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 16, 16, 32)   9248        activation_11[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 16, 16, 32)   128         conv2d_13[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 16, 16, 32)   0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 16, 16, 32)   9248        activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 16, 16, 32)   128         conv2d_14[0][0]                  
__________________________________________________________________________________________________
add_6 (Add)                     (None, 16, 16, 32)   0           activation_11[0][0]              
                                                                 batch_normalization_13[0][0]     
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 16, 16, 32)   0           add_6[0][0]                      
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 8, 8, 64)     18496       activation_13[0][0]              
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 8, 8, 64)     256         conv2d_15[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 8, 8, 64)     0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 8, 8, 64)     36928       activation_14[0][0]              
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 8, 8, 64)     2112        activation_13[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 8, 8, 64)     256         conv2d_16[0][0]                  
__________________________________________________________________________________________________
add_7 (Add)                     (None, 8, 8, 64)     0           conv2d_17[0][0]                  
                                                                 batch_normalization_15[0][0]     
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 8, 8, 64)     0           add_7[0][0]                      
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 8, 8, 64)     36928       activation_15[0][0]              
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 8, 8, 64)     256         conv2d_18[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 8, 8, 64)     0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 8, 8, 64)     36928       activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 8, 8, 64)     256         conv2d_19[0][0]                  
__________________________________________________________________________________________________
add_8 (Add)                     (None, 8, 8, 64)     0           activation_15[0][0]              
                                                                 batch_normalization_17[0][0]     
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 8, 8, 64)     0           add_8[0][0]                      
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 8, 8, 64)     36928       activation_17[0][0]              
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 8, 8, 64)     256         conv2d_20[0][0]                  
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 8, 8, 64)     0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 8, 8, 64)     36928       activation_18[0][0]              
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 8, 8, 64)     256         conv2d_21[0][0]                  
__________________________________________________________________________________________________
add_9 (Add)                     (None, 8, 8, 64)     0           activation_17[0][0]              
                                                                 batch_normalization_19[0][0]     
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 8, 8, 64)     0           add_9[0][0]                      
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 1, 1, 64)     0           activation_19[0][0]              
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 64)           0           average_pooling2d_1[0][0]        
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           650         flatten_1[0][0]                  
==================================================================================================
Total params: 274,442
Trainable params: 273,066
Non-trainable params: 1,376
__________________________________________________________________________________________________
ResNet20v1
Using real-time data augmentation.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.

Epoch 1/50
1563/1563 [==============================] - 63s 41ms/step - loss: 1.5747 - acc: 0.4899 - val_loss: 2.4372 - val_acc: 0.4211

Epoch 00001: val_acc improved from -inf to 0.42110, saving model to /content/saved_models/cifar10_ResNet20v1_model.001.h5
Epoch 2/50
1563/1563 [==============================] - 56s 36ms/step - loss: 1.1636 - acc: 0.6574 - val_loss: 1.3542 - val_acc: 0.6028

Epoch 00002: val_acc improved from 0.42110 to 0.60280, saving model to /content/saved_models/cifar10_ResNet20v1_model.002.h5
Epoch 3/50
1563/1563 [==============================] - 56s 36ms/step - loss: 1.0164 - acc: 0.7153 - val_loss: 1.3679 - val_acc: 0.6091

Epoch 00003: val_acc improved from 0.60280 to 0.60910, saving model to /content/saved_models/cifar10_ResNet20v1_model.003.h5
Epoch 4/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.9191 - acc: 0.7481 - val_loss: 1.2876 - val_acc: 0.6529

Epoch 00004: val_acc improved from 0.60910 to 0.65290, saving model to /content/saved_models/cifar10_ResNet20v1_model.004.h5
Epoch 5/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.8398 - acc: 0.7754 - val_loss: 1.1390 - val_acc: 0.6913

Epoch 00005: val_acc improved from 0.65290 to 0.69130, saving model to /content/saved_models/cifar10_ResNet20v1_model.005.h5
Epoch 6/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.7805 - acc: 0.7953 - val_loss: 0.8066 - val_acc: 0.7926

Epoch 00006: val_acc improved from 0.69130 to 0.79260, saving model to /content/saved_models/cifar10_ResNet20v1_model.006.h5
Epoch 7/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.7313 - acc: 0.8120 - val_loss: 0.9398 - val_acc: 0.7533

Epoch 00007: val_acc did not improve from 0.79260
Epoch 8/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.6930 - acc: 0.8210 - val_loss: 0.8556 - val_acc: 0.7788

Epoch 00008: val_acc did not improve from 0.79260
Epoch 9/50
1563/1563 [==============================] - 55s 35ms/step - loss: 0.6660 - acc: 0.8320 - val_loss: 0.6917 - val_acc: 0.8245

Epoch 00009: val_acc improved from 0.79260 to 0.82450, saving model to /content/saved_models/cifar10_ResNet20v1_model.009.h5
Epoch 10/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.6354 - acc: 0.8380 - val_loss: 0.7638 - val_acc: 0.7991

Epoch 00010: val_acc did not improve from 0.82450
Epoch 11/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.6049 - acc: 0.8487 - val_loss: 0.7793 - val_acc: 0.8024

Epoch 00011: val_acc did not improve from 0.82450
Epoch 12/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.5851 - acc: 0.8545 - val_loss: 0.6949 - val_acc: 0.8209

Epoch 00012: val_acc did not improve from 0.82450
Epoch 13/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.5591 - acc: 0.8625 - val_loss: 0.7824 - val_acc: 0.8004

Epoch 00013: val_acc did not improve from 0.82450
Epoch 14/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.5430 - acc: 0.8675 - val_loss: 0.7185 - val_acc: 0.8171

Epoch 00014: val_acc did not improve from 0.82450
Epoch 15/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.5243 - acc: 0.8718 - val_loss: 0.5764 - val_acc: 0.8557

Epoch 00015: val_acc improved from 0.82450 to 0.85570, saving model to /content/saved_models/cifar10_ResNet20v1_model.015.h5
Epoch 16/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.5096 - acc: 0.8755 - val_loss: 0.5975 - val_acc: 0.8505

Epoch 00016: val_acc did not improve from 0.85570
Epoch 17/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.4964 - acc: 0.8807 - val_loss: 0.6054 - val_acc: 0.8509

Epoch 00017: val_acc did not improve from 0.85570
Epoch 18/50
1563/1563 [==============================] - 55s 35ms/step - loss: 0.4812 - acc: 0.8861 - val_loss: 0.6234 - val_acc: 0.8459

Epoch 00018: val_acc did not improve from 0.85570
Epoch 19/50
1563/1563 [==============================] - 55s 35ms/step - loss: 0.4686 - acc: 0.8886 - val_loss: 0.6084 - val_acc: 0.8460

Epoch 00019: val_acc did not improve from 0.85570
Epoch 20/50
1563/1563 [==============================] - 55s 35ms/step - loss: 0.4575 - acc: 0.8920 - val_loss: 0.6314 - val_acc: 0.8422

Epoch 00020: val_acc did not improve from 0.85570
Epoch 21/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.4465 - acc: 0.8958 - val_loss: 0.5514 - val_acc: 0.8620

Epoch 00021: val_acc improved from 0.85570 to 0.86200, saving model to /content/saved_models/cifar10_ResNet20v1_model.021.h5
Epoch 22/50
1563/1563 [==============================] - 55s 35ms/step - loss: 0.4351 - acc: 0.8977 - val_loss: 0.5809 - val_acc: 0.8576

Epoch 00022: val_acc did not improve from 0.86200
Epoch 23/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.4228 - acc: 0.9021 - val_loss: 0.5701 - val_acc: 0.8615

Epoch 00023: val_acc did not improve from 0.86200
Epoch 24/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.4175 - acc: 0.9031 - val_loss: 0.5397 - val_acc: 0.8658

Epoch 00024: val_acc improved from 0.86200 to 0.86580, saving model to /content/saved_models/cifar10_ResNet20v1_model.024.h5
Epoch 25/50
1563/1563 [==============================] - 55s 35ms/step - loss: 0.4052 - acc: 0.9064 - val_loss: 0.5477 - val_acc: 0.8655

Epoch 00025: val_acc did not improve from 0.86580
Epoch 26/50
1563/1563 [==============================] - 57s 36ms/step - loss: 0.4018 - acc: 0.9082 - val_loss: 0.5341 - val_acc: 0.8696

Epoch 00026: val_acc improved from 0.86580 to 0.86960, saving model to /content/saved_models/cifar10_ResNet20v1_model.026.h5
Epoch 27/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.3932 - acc: 0.9118 - val_loss: 0.5463 - val_acc: 0.8673

Epoch 00027: val_acc did not improve from 0.86960
Epoch 28/50
1563/1563 [==============================] - 56s 36ms/step - loss: 0.3847 - acc: 0.9113 - val_loss: 0.5697 - val_acc: 0.8603

Epoch 00028: val_acc did not improve from 0.86960
Epoch 29/50
1563/1563 [==============================] - 57s 37ms/step - loss: 0.3772 - acc: 0.9138 - val_loss: 0.5693 - val_acc: 0.8642

Epoch 00029: val_acc did not improve from 0.86960
Epoch 30/50
1563/1563 [==============================] - 55s 35ms/step - loss: 0.3657 - acc: 0.9189 - val_loss: 0.5846 - val_acc: 0.8548

Epoch 00030: val_acc did not improve from 0.86960
Epoch 31/50
1563/1563 [==============================] - 55s 35ms/step - loss: 0.3612 - acc: 0.9186 - val_loss: 0.5355 - val_acc: 0.8684

Epoch 00031: val_acc did not improve from 0.86960
Epoch 32/50
1563/1563 [==============================] - 55s 35ms/step - loss: 0.3596 - acc: 0.9194 - val_loss: 0.5083 - val_acc: 0.8786

Epoch 00032: val_acc improved from 0.86960 to 0.87860, saving model to /content/saved_models/cifar10_ResNet20v1_model.032.h5
Epoch 33/50
1563/1563 [==============================] - 54s 35ms/step - loss: 0.3520 - acc: 0.9216 - val_loss: 0.5126 - val_acc: 0.8770

Epoch 00033: val_acc did not improve from 0.87860
Epoch 34/50
1563/1563 [==============================] - 54s 35ms/step - loss: 0.3433 - acc: 0.9249 - val_loss: 0.5325 - val_acc: 0.8686

Epoch 00034: val_acc did not improve from 0.87860
Epoch 35/50
1563/1563 [==============================] - 54s 35ms/step - loss: 0.3370 - acc: 0.9251 - val_loss: 0.5206 - val_acc: 0.8746

Epoch 00035: val_acc did not improve from 0.87860
Epoch 36/50
1563/1563 [==============================] - 54s 35ms/step - loss: 0.3352 - acc: 0.9261 - val_loss: 0.5277 - val_acc: 0.8717

Epoch 00036: val_acc did not improve from 0.87860
Epoch 37/50
1563/1563 [==============================] - 54s 34ms/step - loss: 0.3311 - acc: 0.9279 - val_loss: 0.5086 - val_acc: 0.8770

Epoch 00037: val_acc did not improve from 0.87860
Epoch 38/50
1563/1563 [==============================] - 54s 35ms/step - loss: 0.3223 - acc: 0.9307 - val_loss: 0.5261 - val_acc: 0.8779

Epoch 00038: val_acc did not improve from 0.87860
Epoch 39/50
1563/1563 [==============================] - 54s 35ms/step - loss: 0.3180 - acc: 0.9328 - val_loss: 0.5364 - val_acc: 0.8721

Epoch 00039: val_acc did not improve from 0.87860
Epoch 40/50
1563/1563 [==============================] - 54s 34ms/step - loss: 0.3126 - acc: 0.9334 - val_loss: 0.4831 - val_acc: 0.8843

Epoch 00040: val_acc improved from 0.87860 to 0.88430, saving model to /content/saved_models/cifar10_ResNet20v1_model.040.h5
Epoch 41/50
1563/1563 [==============================] - 54s 34ms/step - loss: 0.3093 - acc: 0.9338 - val_loss: 0.6002 - val_acc: 0.8604

Epoch 00041: val_acc did not improve from 0.88430
Epoch 42/50
1563/1563 [==============================] - 54s 34ms/step - loss: 0.3049 - acc: 0.9356 - val_loss: 0.5094 - val_acc: 0.8828

Epoch 00042: val_acc did not improve from 0.88430
Epoch 43/50
1563/1563 [==============================] - 53s 34ms/step - loss: 0.3004 - acc: 0.9349 - val_loss: 0.5185 - val_acc: 0.8753

Epoch 00043: val_acc did not improve from 0.88430
Epoch 44/50
1563/1563 [==============================] - 54s 35ms/step - loss: 0.2942 - acc: 0.9365 - val_loss: 0.4900 - val_acc: 0.8894

Epoch 00044: val_acc improved from 0.88430 to 0.88940, saving model to /content/saved_models/cifar10_ResNet20v1_model.044.h5
Epoch 45/50
1563/1563 [==============================] - 54s 35ms/step - loss: 0.2923 - acc: 0.9387 - val_loss: 0.5064 - val_acc: 0.8833

Epoch 00045: val_acc did not improve from 0.88940
Epoch 46/50
1563/1563 [==============================] - 54s 34ms/step - loss: 0.2877 - acc: 0.9400 - val_loss: 0.5980 - val_acc: 0.8632

Epoch 00046: val_acc did not improve from 0.88940
Epoch 47/50
1563/1563 [==============================] - 54s 34ms/step - loss: 0.2858 - acc: 0.9406 - val_loss: 0.5139 - val_acc: 0.8797

Epoch 00047: val_acc did not improve from 0.88940
Epoch 48/50
1563/1563 [==============================] - 53s 34ms/step - loss: 0.2836 - acc: 0.9419 - val_loss: 0.4917 - val_acc: 0.8866

Epoch 00048: val_acc did not improve from 0.88940
Epoch 49/50
1563/1563 [==============================] - 53s 34ms/step - loss: 0.2767 - acc: 0.9428 - val_loss: 0.4870 - val_acc: 0.8842

Epoch 00049: val_acc did not improve from 0.88940
Epoch 50/50
1563/1563 [==============================] - 54s 35ms/step - loss: 0.2721 - acc: 0.9453 - val_loss: 0.4751 - val_acc: 0.8933

Epoch 00050: val_acc improved from 0.88940 to 0.89330, saving model to /content/saved_models/cifar10_ResNet20v1_model.050.h5
10000/10000 [==============================] - 2s 150us/step
Test loss: 0.4750881087779999
Test accuracy: 0.8933
