#-------------------------------------------------------------------------------
# Name:        Deep Convolutional Neural Network (DCNN) for slum mapping
# Purpose:     Test DCNN on mapping degrees of deprivation
#              Main method is ...

# Version:     0.1 
#              Functionalities:
#              1.
#              2.
#              3.
#              4.

# Author:      Jiong (Jon) Wang
#
# Created:     26/05/2020
# Copyright:   (c) JonWang 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------


from keras import backend as K
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers import (
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Input,
    Conv2DTranspose,
    Flatten,
    BatchNormalization,
    Activation,
    concatenate
)
from keras.callbacks import TensorBoard 
from keras.layers.core import RepeatVector, Reshape
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage import measure
from skimage.util.shape import view_as_blocks
from imgaug import augmenters as iaa
from scipy import interpolate
from gdal import Open, GA_Update
from osgeo import gdal_array
from pathlib import Path
from PIL import Image
import sklearn.feature_extraction.image as im
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import tensorflow as tf


##############################################
# 01 Load and organize images
##############################################

# load rasters: base image along with its label
# Stack base image and label into a list of array
def load_rasters(path, subUL, extent):  # Subset from original raster with extent and upperleft coord
    """Load training data pairs (two high resolution images and two low resolution images)"""
    path_list = []  # List image name and path
    for path in Path(path).glob('*.tif'):
        path_list.append(path)
    assert len(path_list) == 2

#    for path in path_list:  # Organize path list
#        img_name = path.name
#        if 'base' in img_name:
#            base = path
#        elif 'label' in img_name:
#            label = path
#    path_list = [base, label]
    
    stack = []  # Stack base and label together into a 3D array
    for path in path_list:
        if path.name.startswith('base'):
            data = gdal_array.LoadFile(str(path), xoff=subUL[0], yoff=subUL[1], 
                                       xsize=extent, ysize=extent) #.astype(np.int)
            for band in range(data.size[2]):
                stack.append(band)
        else:
            data = gdal_array.LoadFile(str(path), xoff=subUL[0], yoff=subUL[1], 
                                       xsize=extent, ysize=extent) #.astype(np.int)
#        image = Image.fromarray(data)
#        data = nan_remover(data)
#        setattr(image, 'filename', path)
        stack.append(data)
    # Ensure the size of base and label is are consistent
    assert stack[0].size == stack[-1].size
    return stack[:-1], stack[-1]


# Clean the NaN values
def nan_remover(array):
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # Masking invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # Getting only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    array_interp = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy), method='nearest')
    # Clean the edge
    bad_indexes = np.isnan(array_interp)
    good_indexes = np.logical_not(bad_indexes)
    good_data = array_interp[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    array_interp[bad_indexes] = interpolated
    return array_interp


##############################################
# 02 Prepare data for train, val, test
##############################################

# Sample patches from an image band/layer
# Stride controls the overlap of patches
def gen_patches(image, size, stride):
    """Segment input images into patches"""
    if not isinstance(size, tuple):  # Ensure format is tuple
        size = (size, size)
    if stride is None:
        stride = size
    elif not isinstance(stride, tuple):
        stride = (stride, stride)
    # Columns in priority
    for i in range(0, image.shape[0] - size[0] + 1, stride[0]):  # One patch every stride
        for j in range(0, image.shape[1] - size[1] + 1, stride[1]):
            yield image[i:i + size[0], j:j + size[1], np.newaxis]  # If Pillow Image is used: image.crop([i, j, i + size[0], j + size[1]])


# Advanced version of patch sampling
# Sample patches at target areas with criteria, such as label size
def gen_patches_ctrl():  # With controlled position
    
    return None


# Generate patches for all layers/bands in stack
def stack_to_patches(stack, size, stride, patches):
#    assert len(stack) == 4
    for i in range(len(stack)):  # Loop over the layers/bands in the stack
        patches[i] += [img for img in gen_patches(stack[i], size, stride)]  # If Pillow Image: img_to_array(img)


# Arrange training and validation sets from the patches
def load_train_set(data_dir, subUL, extent, size, stride=2):
    # Load image data from training folder
    patches = [[] for _ in range(4)]  # Empty list to store patches for each layer/band in stack
    for path in (data_dir/'Train').glob('*'):  # Loop over all folders
        if path.is_dir():
            print('loading image pairs from {}'.format(path))
            stack = load_rasters(path, subUL, extent)
            stack = [*stack[0], stack[1]]
            # subset samples into patches
            stack_to_patches(stack, size, stride, patches)
            
    # Split patches into training and validation sets        
    patch_train = [[] for _ in range(4)]
    patch_val = [[] for _ in range(4)]
    for i in range(4):
        patch_train[i] = np.stack(patches[i][:int(len(patches[i])*0.7)])
        patch_val[i] = np.stack(patches[i][int(len(patches[i])*0.7):])
    # Return 4-dimensional array (number, height, width, channel)
    return patch_train[:3], patch_train[-1], patch_val[:3], patch_val[-1]


# Arrange test set by using another set of raster input
def load_test_set(stack, block_size=(10, 10)):
    assert len(stack) == 2

    patches = [[] for _ in range(4)]
    stack = [*stack[0], stack[1]]
    stack_to_patches(stack, size=tuple(block_size), patches)

    for i in range(4):
        patches[i] = np.stack(patches[i])
    return patches[:3], patches[-1]


##############################################
# 03 Customize loss function metrics
##############################################

# Jaccard index realized as intersection over union (iou)
def mean_iou(y_true, y_pred):
    prec = []
    # Consider y_pred larger than 0.5
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# Covariance
def cov(y_true, y_pred):
    return K.mean((y_true - K.mean(y_true)) * K.transpose((y_pred - K.mean(y_pred))))


# Correlation
def r2(y_true, y_pred):
    # mean calls tensor property instead of ndarray
    tf_true = y_true
    if not isinstance(y_true, tf.Tensor):
        tf_true = tf.convert_to_tensor(y_true)
    res = K.sum(K.square(y_true - y_pred))
    tot = K.sum(K.square(y_true - K.mean(tf_true)))
    return 1 - res / (tot + K.epsilon())


# Signal-to-noise ratio
def psnr(y_true, y_pred, data_range=50):
    #Peak signal-to-noise ratio averaged over samples and channels
    mse = K.mean(K.square(y_true - y_pred), axis=(-3, -2))
    return K.mean(20 * K.log(data_range / K.sqrt(mse)) / np.log(10))


# structural similarity measurement system
def ssim(y_true, y_pred, data_range=50):
    """structural similarity measurement system."""
    K1 = 0.01
    K2 = 0.03

    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)

    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = cov(y_true, y_pred)

    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    return ((2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) /
            (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))


##############################################
# 04 Convolutional Neural Networks
##############################################

# U-net 
def u_net(img):
    inputs = Input(shape=img.shape[-3:])
    conv1_1 = Conv2D(16, (3, 3), padding='same')(inputs)
    bn1_1 = BatchNormalization(axis=3)(conv1_1)
    relu1_1 = Activation('relu')(bn1_1)
    conv1_2 = Conv2D(16, (3, 3), padding='same')(relu1_1)
    bn1_2 = BatchNormalization(axis=3)(conv1_2)
    relu1_2 = Activation('relu')(bn1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(relu1_2)
    
    conv2_1 = Conv2D(32, (3, 3), padding='same')(pool1)
    bn2_1 = BatchNormalization(axis=3)(conv2_1)
    relu2_1 = Activation('relu')(bn2_1)
    conv2_2 = Conv2D(32, (3, 3), padding='same')(relu2_1)
    bn2_2 = BatchNormalization(axis=3)(conv2_2)
    relu2_2 = Activation('relu')(bn2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(relu2_2)
    
    conv3_1 = Conv2D(64, (3, 3), padding='same')(pool2)
    bn3_1 = BatchNormalization(axis=3)(conv3_1)
    relu3_1 = Activation('relu')(bn3_1)
    conv3_2 = Conv2D(64, (3, 3), padding='same')(relu3_1)
    bn3_2 = BatchNormalization(axis=3)(conv3_2)
    relu3_2 = Activation('relu')(bn3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(relu3_2)
    
    conv4_1 = Conv2D(128, (3, 3), padding='same')(pool3)
    bn4_1 = BatchNormalization(axis=3)(conv4_1)
    relu4_1 = Activation('relu')(bn4_1)
    conv4_2 = Conv2D(128, (3, 3), padding='same')(relu4_1)
    bn4_2 = BatchNormalization(axis=3)(conv4_2)
    relu4_2 = Activation('relu')(bn4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(relu4_2)
    
    conv5_1 = Conv2D(256, (3, 3), padding='same')(pool4)
    bn5_1 = BatchNormalization(axis=3)(conv5_1)
    relu5_1 = Activation('relu')(bn5_1)
    conv5_2 = Conv2D(256, (3, 3), padding='same')(relu5_1)
    bn5_2 = BatchNormalization(axis=3)(conv5_2)
    relu5_2 = Activation('relu')(bn5_2)
    
    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(relu5_2), relu4_2], axis=3)
    conv6_1 = Conv2D(128, (3, 3), padding='same')(up6)
    bn6_1 = BatchNormalization(axis=3)(conv6_1)
    relu6_1 = Activation('relu')(bn6_1)
    conv6_2 = Conv2D(128, (3, 3), padding='same')(relu6_1)
    bn6_2 = BatchNormalization(axis=3)(conv6_2)
    relu6_2 = Activation('relu')(bn6_2)
    
    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(relu6_2), relu3_2], axis=3)
    conv7_1 = Conv2D(64, (3, 3), padding='same')(up7)
    bn7_1 = BatchNormalization(axis=3)(conv7_1)
    relu7_1 = Activation('relu')(bn7_1)
    conv7_2 = Conv2D(64, (3, 3), padding='same')(relu7_1)
    bn7_2 = BatchNormalization(axis=3)(conv7_2)
    relu7_2 = Activation('relu')(bn7_2)
    
    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(relu7_2), relu2_2], axis=3)
    conv8_1 = Conv2D(32, (3, 3), padding='same')(up8)
    bn8_1 = BatchNormalization(axis=3)(conv8_1)
    relu8_1 = Activation('relu')(bn8_1)
    conv8_2 = Conv2D(32, (3, 3), padding='same')(relu8_1)
    bn8_2 = BatchNormalization(axis=3)(conv8_2)
    relu8_2 = Activation('relu')(bn8_2)
    
    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(relu8_2), relu1_2], axis=3)
    conv9_1 = Conv2D(16, (3, 3), padding='same')(up9)
    bn9_1 = BatchNormalization(axis=3)(conv9_1)
    relu9_1 = Activation('relu')(bn9_1)
    conv9_2 = Conv2D(16, (3, 3), padding='same')(relu9_1)
    bn9_2 = BatchNormalization(axis=3)(conv9_2)
    relu9_2 = Activation('relu')(bn9_2)
    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(relu9_2)
    
    model = Model(inputs=[inputs], outputs=[conv10])
    print(model.summary())
    
    return model


def get_model(name):
    """Get model function from the name space in strings"""
    return globals()[name]


##############################################
# 05 Train and predict
##############################################

# Define an experiment for training and test session
class Experiment(object):
    def __init__(self, load_set, build_model, optimizer, save_dir='.'):
        self.load_set = load_set
        self.build_model = build_model
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.save_dir / 'config.yaml'
        self.model_file = self.save_dir / 'model.hdf5'
        self.visual_file = self.save_dir / 'model.eps'

        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history_file = self.train_dir / 'history.csv'
        self.weights_dir = self.train_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)

        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)

    def weights_file(self, epoch=None):
        if epoch is None:
            return self.weights_dir / 'ep{epoch:04d}.hdf5'
        else:
            return self.weights_dir / 'ep{:04d}.hdf5'.format(epoch)

    @property
    def latest_epoch(self):
        try:
            return pd.read_csv(str(self.history_file))['epoch'].iloc[-1]
        except (FileNotFoundError, pd.io.common.EmptyDataError):
            pass
        return -1

    @staticmethod
    def _ensure_dimension(array, dim):
        while len(array.shape) < dim:
            array = array[np.newaxis, ...]
        return array

    @staticmethod
    def _ensure_channel(array, c):
        return array[..., c:c + 1]

    @staticmethod
    def validate(array):
        array = Experiment._ensure_dimension(array, 4)
        array = Experiment._ensure_channel(array, 0)
        return array

    def compile(self, model):
        """Compile model with default settings."""
        model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=[mean_iou])
        return model

    def train(self, data_dir, epochs, resume=True):
        # Load and process data
        x_train, y_train, x_val, y_val = self.load_set(data_dir)
        assert len(x_train) == 3 and len(x_val) == 3
        for i in range(3):
            x_train[i], x_val[i] = [self.validate(x) for x in [x_train[i], x_val[i]]]
        y_train, y_val = [self.validate(y) for y in [y_train, y_val]]

        # Compile model
        model = self.compile(self.build_model(*x_train))
        model.summary()
        #self.config_file.write_text(model.to_yaml())
        #plot_model(model, to_file=str(self.visual_file), show_shapes=False)

        # Inherit weights
        if resume:
            latest_epoch = self.latest_epoch
            if latest_epoch > -1:
                weights_file = self.weights_file(epoch=latest_epoch)
                model.load_weights(str(weights_file))
            initial_epoch = latest_epoch + 1
        else:
            initial_epoch = 0

        # Set up callbacks
        callbacks = []
        callbacks += [ModelCheckpoint(str(self.model_file))]
        callbacks += [ModelCheckpoint(str(self.weights_file()),
                                      save_weights_only=True)]
        callbacks += [CSVLogger(str(self.history_file), append=resume)]

        # Train
        model.fit(x_train, y_train, batch_size=16, epochs=epochs, callbacks=callbacks,
                  validation_data=(x_val, y_val), initial_epoch=initial_epoch)

        # Plot metrics history
        prefix = str(self.history_file).rsplit('.', maxsplit=1)[0]
        df = pd.read_csv(str(self.history_file))
        epoch = df['epoch']
        for metric in ['Loss', 'PSNR', 'R2']:
            train = df[metric.lower()]
            val = df['val_' + metric.lower()]
            plt.figure()
            plt.plot(epoch, train, label='train')
            plt.plot(epoch, val, label='val')
            plt.legend(loc='best')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.savefig('.'.join([prefix, metric.lower(), 'eps']))
            plt.close()

    def test(self, data_dir, test_set='Test', block_size=(10, 10), metrics=[mean_iou]):
        print('Testing...')
        output_dir = self.test_dir / test_set
        output_dir.mkdir(exist_ok=True)

        # Evaluate metrics on each image
        rows = []
        for image_path in (data_dir / test_set).glob('*'):
            if image_path.is_dir():
                rows += [self.test_on_image(image_path, output_dir, block_size=block_size, metrics=metrics)]
        df = pd.DataFrame(rows)
        # Compute average metrics
        row = pd.Series()
        row['name'] = 'average'
        for col in df:
            if col != 'name':
                row[col] = df[col].mean()
        df = df.append(row, ignore_index=True)
        df.to_csv(str(self.test_dir / '{}/metrics.csv'.format(test_set)))

    def test_on_image(self, image_dir, output_dir, block_size=(10, 10), metrics=[mean_iou]):
        # Load images
        print('Loading image pairs from {}'.format(image_dir))
        input_images, valid_image = load_rasters(image_dir, subUL, extent)
        assert len(input_images) == 3
        name = input_images[-1].filename.name if hasattr(input_images[-1], 'filename') else ''
        print('Predict on image {}'.format(name))

        # Generate output image and measure run time
        # The shape of the x_inputs (numbers, height, width, channels)
        x_inputs = [self.validate(img_to_array(im)) for im in input_images]
        assert x_inputs[0].shape[1] % block_size[0] == 0
        assert x_inputs[0].shape[2] % block_size[1] == 0
        x_train, _ = load_test_set((input_images, valid_image),
                                   block_size=block_size, scale=self.scale)

        model = self.compile(self.build_model(*x_train))
        if self.model_file.exists():
            model.load_weights(str(self.model_file))

        t_start = time.perf_counter()
        y_preds = model.predict(x_train, batch_size=1)  # 4-dimensional array with batch size
        # map predicted patches back to original image extent
        y_pred = np.empty(x_inputs[1].shape[-3:], dtype=np.float32)
        row_step = block_size[0]
        col_step = block_size[1]
        rows = x_inputs[0].shape[2] // block_size[1]
        cols = x_inputs[0].shape[1] // block_size[0]
        count = 0
        for j in range(rows):
            for i in range(cols):
                y_pred[i * row_step: (i + 1) * row_step, j * col_step: (j + 1) * col_step] = y_preds[count]
                count += 1
        assert count == rows * cols
        t_end = time.perf_counter()

        # Record metrics
        row = pd.Series()
        row['name'] = name
        row['time'] = t_end - t_start
        y_true = self.validate(img_to_array(valid_image))
        y_pred = self.validate(y_pred)
        for metric in metrics:
            row[metric.__name__] = K.eval(metric(y_true, y_pred))

        prototype = str(valid_image.filename) if hasattr(valid_image, 'filename') else None
        gdal_array.SaveArray(y_pred[0].squeeze().astype(np.int16),
                             str(output_dir / name),
                             prototype=prototype)
        return row


##############################################
# Main
##############################################

def main():

#    parser = argparse.ArgumentParser()
#    parser.add_argument('config', type=Path)
#    args = parser.parse_args()
#    param = json.load(args.config.open())
    with open('dcfnex-tir.json', 'r') as read_file:
        param = json.load(read_file)

    repo_dir = Path('__file__').parents[0]
    data_dir = repo_dir / 'TrainingData'
    
    # Input training patch dimensions
    img_rows, img_cols = 32, 32
    # The images are 3-channel (RGB).
    img_channels = 3
    
    # Subset study area
    subUL = [, ]
    extent = 
        
#    input_suffix = 'input'
#    pred_suffix = 'pred'
#    valid_suffix = 'valid'
#    
#    modis_prefix = 'MOD11A1'
#    landsat_prefix = 'LC08'
    
    # Model
    build_model = partial(get_model(param['model']['name']),
                          **param['model']['params'])
    if 'optimizer' in param:
        optimizer = getattr(optimizers, param['optimizer']['name'].lower())
        optimizer = optimizer(**param['optimizer']['params'])
    else:
        optimizer = 'adam'
    
    lr_block_size = tuple(param['lr_block_size'])
    
    load_train_set = partial(load_train_set,
                             lr_sub_size=param['lr_sub_size'],
                             lr_sub_stride=param['lr_sub_stride'],
                             ls_subUL=ls_subUL, coarse_resol=coarse_resol,
                             fine_resol=fine_resol, mod_subUL=mod_subUL, extent=extent)
        
    # Training
    expt = Experiment(scale=param['scale'], load_set=load_train_set,
                      build_model=build_model, optimizer=optimizer,
                      save_dir=param['save_dir'])
    print('training process...')
    expt.train(train_set=data_dir, epochs=param['epochs'], resume=True)
    
    # Evaluation
    print('evaluation process...')
    test_set = 'Test'
    expt.test(data_dir, test_set=test_set, lr_block_size=[64, 64])  # lr_block_size=lr_block_size
#    for test_set in param['test_sets']:
#        expt.test(test_set=test_set, lr_block_size=lr_block_size)


##############################################
# Execution
##############################################
if __name__ == '__main__':
    main()
# Define input image dimensions
img_rows, img_cols = 32, 32
# The images are RGB.
img_channels = 3
