"""
Model definition:
    Compound architecture composed of 4 components;
    Encoder, Decoder, Fusion and InceptionResNetV2;
    Manipulated from deepColorization.

Jon Wang, 2018 June 6.
"""


from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers import (
    Conv2D,
    UpSampling2D,
    Input,
    Conv2DTranspose,
    Flatten,
    Activation,
    concatenate
)
from keras.callbacks import TensorBoard 
from keras.layers.core import RepeatVector, Reshape
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage import measure
from skimage.io import imsave
from skimage.util.shape import view_as_blocks
import sklearn.feature_extraction.image as im
from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt
import numpy as np
from gdal import Open, GA_Update
import sys
import os
from keras import backend as K
import tensorflow as tf


# ========== INPUT ==========
# Define input image dimensions
img_rows, img_cols = 32, 32
# The images are RGB.
img_channels = 3

# Use GDAL OPEN to read base RGB WorldView(WV02) map and corresponding reference label
wv = Open('mapMul.tif')
land = np.zeros((wv.RasterYSize,wv.RasterXSize,3)).astype('uint16')  #np.zeros_like(wv).astype('int16')
land[:,:,0] = wv.GetRasterBand(5).ReadAsArray()  # swaping the band and re-arrange the band order into 1-R, 2-G and 3-B
land[:,:,1] = wv.GetRasterBand(3).ReadAsArray()
land[:,:,2] = wv.GetRasterBand(2).ReadAsArray()
del wv  # delete variables to save memory resources
label = imread("labelMul.tif")

wv2 = Open('mapMul2.tif')
land2 = np.zeros((wv2.RasterYSize,wv2.RasterXSize,3)).astype('uint16')  #np.zeros_like(wv).astype('int16')
land2[:,:,0] = wv2.GetRasterBand(5).ReadAsArray()  # swaping the band and re-arrange the band order into 1-R, 2-G and 3-B
land2[:,:,1] = wv2.GetRasterBand(3).ReadAsArray()
land2[:,:,2] = wv2.GetRasterBand(2).ReadAsArray()
del wv2  # delete variables to save memory resources

# Convert to 0-255 valued UINT8 image
land = cv2.convertScaleAbs(land,alpha = 255.0/(np.max(land)))
land2 = cv2.convertScaleAbs(land2,alpha = 255.0/(np.max(land2)))
slum = (label > 0).astype('uint8')
slum2 = np.zeros((land2.shape[0],land2.shape[1])).astype('uint8')
del label  # delete variables to save memory resources
#plt.imshow(slum, 'gray')
land_slum = np.dstack((land, slum))  # stack the map and label for drawing training patches
land_slum2 = np.dstack((land2, slum2))

# Sample from slum blocks according to the size of connected slum blocks
slum_block = measure.label(slum, background = False, connectivity = 1)  # extract and label connected slum blocks
regions = measure.regionprops(slum_block)  # get the properties of labelled slum blocks
slum_areas = []  # store the size of area of labelled slum blocks for sorting and choosing large slum blocks as samples

for region in regions:  # query the slum block area property and store the sizes
    slum_areas.append(region.area)
sort_index = [b[0] for b in sorted(enumerate(slum_areas),key=lambda i:i[1])]  # sort the sizes of slum blocks

sample_block = 4  # draw samples from several numbers of slum blocks
# slum_patches = 1000  # number of training patches drawn from sample slum blocks 
non_slum_patches = 2000
np.random.get_state()  # create pseudo random state to draw training patches from the bounding sample area
rng = np.random.RandomState(123)

for j in range (sample_block):
    slum_bounding = regions[sort_index[np.shape(slum_areas)[0]-(j+1)]].bbox  # identify the bounding of the largest slum block
    # plt.imshow(land_slum[slum_bounding[0]:slum_bounding[2], slum_bounding[1]:slum_bounding[3], 3])
    # draw randomly sample patches for training and testing
    slum_patches = int((slum_bounding[2]-slum_bounding[0])*(slum_bounding[3]-slum_bounding[1])/5)  # number of training patches drawn from sample slum blocks 
    if j < 1:
        datasets0 = im.extract_patches_2d(land_slum[slum_bounding[0]:slum_bounding[2], slum_bounding[1]:slum_bounding[3], :],(img_rows,img_cols),slum_patches, random_state = rng)
    else:
        datasetsTemp = im.extract_patches_2d(land_slum[slum_bounding[0]:slum_bounding[2], slum_bounding[1]:slum_bounding[3], :],(img_rows,img_cols),slum_patches, random_state = rng)
        datasets0 = np.concatenate((datasets0, datasetsTemp), axis=0)      
datasets1 = im.extract_patches_2d(land_slum[1700:2000, 1990:2200, :],(img_rows,img_cols),non_slum_patches, random_state = rng)
datasets2 = im.extract_patches_2d(land_slum[1600:1850, 3400:3600, :],(img_rows,img_cols),non_slum_patches, random_state = rng)
datasets3 = im.extract_patches_2d(land_slum2[1600:1900, 1000:1300, :],(img_rows,img_cols),non_slum_patches, random_state = rng)
datasets = np.concatenate((datasets0, datasets1, datasets2, datasets3),axis=0)
np.random.shuffle(datasets)  # randomly shuffle the patches
del land_slum, land, slum, land_slum2, land2, slum2, slum_block, datasetsTemp, datasets0, datasets1, datasets2, datasets3  # delete variables to save memory resources

# Image Augmentation !!
# Sometimes(0.8, ...) applies the given augmenter in 80% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
    sometimes(iaa.Crop(px=(0, 1))), # crop images from each side by 0 to 16px (randomly chosen)
    sometimes(iaa.Fliplr(1)), # horizontally flip 50% of the image
    sometimes(iaa.GaussianBlur(sigma=(0, 0.1))), # blur images with a sigma of 0 to .1
    sometimes(iaa.Affine(
                         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                         #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                         rotate=(-45, 45), # rotate by -45 to +45 degrees
                         shear=(-5, 5), # shear by -10 to +10 degrees
                         #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
              )),
    sometimes(iaa.PiecewiseAffine(scale=(0, 0.03))),
    sometimes(iaa.PerspectiveTransform(scale=(0, 0.1)))
])

datasets = seq.augment_images(datasets)
datasets = datasets.astype('float32')
np.random.shuffle(datasets)


# ========== MODEL ==========
#Load InceptionResNet model without weights
# ========== LOAD MODEL ==========
json_file = open('Embed.json','r')
loaded_embed_json = json_file.read()
json_file.close()
Embed = model_from_json(loaded_embed_json)
Embed.load_weights('EmbedWeights.h5')
Embed.graph = tf.get_default_graph()
# Output dimension of the InceptionResNet
embed_input = Input(shape=(256,))

#Encoder
encoder_input = Input(shape=(img_rows, img_cols, img_channels,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#Fusion
fusion_rows, fusion_cols = np.int(img_rows/(2**3)), np.int(img_cols/(2**3))  # size decreases 2**3 after 3 times of striding
fusion_output = RepeatVector(fusion_rows * fusion_cols)(embed_input) 
fusion_output = Reshape(([fusion_rows, fusion_cols, 256]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

#Decoder
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
#decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2DTranspose(128, (3, 3), padding='same', strides=2)(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
#decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2DTranspose(64, (3, 3), padding='same', strides=2)(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2DTranspose(16, (3, 3), padding='same', strides=2)(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoder_output)
#decoder_output = UpSampling2D((2, 2))(decoder_output)
#flatten_output = Reshape((img_rows*img_cols,1))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

# Create embedding
def create_inception_embedding(datasets):
    datasets = datasets[:,:,:,0:img_channels]
    mean_image = np.mean(datasets, axis=0)
    datasets -= mean_image
    datasets /= 128.
    with Embed.graph.as_default():
        embed = Embed.predict(datasets)
    return embed


# ========== TRAIN ========== 
# Generate training data
batch_size = 32

data_augmentation = False

if not data_augmentation:
    print('Not using data augmentation.')
    X_train = datasets[:int(datasets.shape[0]*0.7),:,:,0:img_channels]
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_train /= 128.
    Y_train = datasets[:int(datasets.shape[0]*0.7),:,:,img_channels]
    Y_train = Y_train.reshape(Y_train.shape+(1,))
    emb_train = create_inception_embedding(datasets[:int(datasets.shape[0]*0.7),:,:,:])
    
    X_val = datasets[int(datasets.shape[0]*0.7):datasets.shape[0],:,:,0:img_channels]
    mean_image = np.mean(X_val, axis=0)
    X_val -= mean_image
    X_val /= 128.
    Y_val = datasets[int(datasets.shape[0]*0.7):datasets.shape[0],:,:,img_channels]
    Y_val = Y_val.reshape(Y_val.shape+(1,))
    emb_val = create_inception_embedding(datasets[int(datasets.shape[0]*0.7):datasets.shape[0],:,:,:])
    
    del datasets
    
    tensorboard = TensorBoard(log_dir="/output")
    opt = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=30)
    csv_logger = CSVLogger('slumDetectLog3.csv')
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # loss = 'binary_crossentropy' or 'mse'
    model.fit([X_train, emb_train], Y_train,
              batch_size=batch_size,
              nb_epoch=50,
              validation_data=([X_val, emb_val], Y_val),
              callbacks=[lr_reducer, early_stopper, csv_logger])  # or callbacks=[tensorboard]
    
    model.save_weights('slumDetecWeights3.h5')

else:
    print('Using real-time data augmentation.')
    # Image transformer
    datagen = ImageDataGenerator(
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range=40,
            horizontal_flip=True)
    
    def batch_gen(datasets, batch_size):
        for batch in datagen.flow(datasets, batch_size=batch_size):
            embed = create_inception_embedding(batch)
            X_train = batch[:,:,:,0:img_channels]
            mean_image = np.mean(X_train, axis=0)
            X_train -= mean_image
            X_train /= 128.            
            Y_train = batch[:,:,:,img_channels]
            Y_train = Y_train.reshape(Y_train.shape+(1,))
            yield ([X_train, embed], Y_train)
    
    tensorboard = TensorBoard(log_dir="/output")
    opt = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('slumDetectLog3.csv')
    model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics= ['accuracy'])  # or loss = 'mse'
    model.fit_generator(batch_gen(datasets[:int(datasets.shape[0]*0.8),:,:,:], batch_size),
                                  steps_per_epoch=datasets[:int(datasets.shape[0]*0.8),:,:,:].shape[0]//batch_size+1,
                                  epochs=500,
                                  validation_data=batch_gen(datasets[int(datasets.shape[0]*0.8):datasets.shape[0],:,:,:], batch_size),
                                  validation_steps=datasets[int(datasets.shape[0]*0.8):datasets.shape[0],:,:,:].shape[0]//batch_size+1,
                                  callbacks=[lr_reducer, early_stopper, csv_logger])  # or callbacks=[tensorboard]
    
    model.save_weights('slumDetecWeights3.h5')


# ========== PREDICTION ==========

# Open a series of map in target folder for prediction
path = ".\MulAll"
filelist = os.listdir(path)

for files in filelist:
    wv = Open(os.path.join(path,files))
    #wv = Open('map.tif')
    pred = np.zeros((wv.RasterYSize,wv.RasterXSize,3)).astype('uint16')  #np.zeros_like(wv).astype('int16')
    pred[:,:,0] = wv.GetRasterBand(5).ReadAsArray()  # swaping the band and re-arrange the band order into 1-R, 2-G and 3-B
    pred[:,:,1] = wv.GetRasterBand(3).ReadAsArray()
    pred[:,:,2] = wv.GetRasterBand(2).ReadAsArray()
    del wv  # delete variables to save memory resources

    # Convert to 0-255 valued UINT8 image
    pred = cv2.convertScaleAbs(pred,alpha = 255.0/(np.max(pred)))
    visualize = pred

    # Pad the image so that it can be divisible for patches with size of (img_rows * img_cols)
    pred_row, pred_col = pred.shape[0], pred.shape[1]
    pred = np.lib.pad(pred, ((0, img_rows-pred_row%img_rows), (0, img_cols-pred_col%img_cols),(0,0)), 'edge')
    
    # Slice the image into blocks with size of (img_rows * img_cols)
    pred = np.reshape(pred, (pred.shape[0], pred.shape[1], pred.shape[2]))
    pred_block = view_as_blocks(pred, block_shape=(img_rows, img_cols, img_channels))
    del pred  # Delete variables to save memory
    row_block = pred_block.shape[0]
    col_block = pred_block.shape[1]
    # Reshape blocks for proper model input
    pred_block = pred_block.reshape(pred_block.shape[0]*pred_block.shape[1],img_rows,img_cols,img_channels)
    pred_block = pred_block.astype('float32')
    
    pred_augmentation = False
    
    if not pred_augmentation:
        print('Not using data augmentation.')
        # Normalize input for prediction
        pred_emb = create_inception_embedding(pred_block)
        mean_image = np.mean(pred_block, axis=0)
        pred_block -= mean_image
        pred_block /= 128.

        
        pred_label = model.predict([pred_block, pred_emb])
        
        # Delete variables to save memory
        del pred_block    
    
    else:
        print('Using real-time data augmentation.')
        # Generate on-the-fly batches
        pred_datagen = ImageDataGenerator()
        def image_pred_gen(pred_block, batch_size):
            for batch in pred_datagen.flow(pred_block, batch_size=batch_size):
                embed = create_inception_embedding(batch)
                X_batch = batch[:,:,:,0:img_channels]
                mean_image = np.mean(X_batch, axis=0)
                X_batch -= mean_image
                X_batch /= 128.           
                yield ([X_batch, embed])
        
        # make prediction
        pred_label = model.predict_generator(image_pred_gen(pred_block, batch_size),
                                             steps=(pred_block.shape[0]//batch_size)+1)
        del pred_block    

    
    # Reshape pred_label into patches
    pred_label = pred_label.reshape(pred_label.shape[0], img_rows, img_cols, pred_label.shape[-1])
    # Create matrix to map label patches to original map size
    label_full = np.zeros((row_block*img_rows, col_block*img_cols))

    for i in range(row_block):
        for j in range(col_block):
            label_full[i*img_rows:(i+1)*img_rows,j*img_cols:(j+1)*img_cols] = pred_label[(i*col_block)+j,:,:,0]       
    # delete variables to save memory
    del pred_label
    
    label_full = label_full[:pred_row,:pred_col]
    np.save('label_pred.npy', label_full)  # save to *.npy file
    
    # Visualization
    #scipy.misc.imsave('slumHeatMap.tif',label_full)
    #plt.imshow(label_full,'Reds')
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1,2,1), plt.imshow(visualize)
    plt.subplot(1,2,2), plt.imshow(label_full)
    #plt.savefig('testResult.tif', dpi=300)

    # !!!! Save GEOreferenced TIFF
    pred = np.load('label_pred.npy')
    pred = (pred*255).astype('uint8')
    #pred = (pred > 0.9).astype('uint8')
    cv2.imwrite(files, pred)
    #cv2.imwrite(files, pred)
    
    dataset = Open(os.path.join(path,files))
    if dataset is None:
        print('Unable to open')
        sys.exit(1)
    
    projection   = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    
    if projection is None and geotransform is None:
        print('No projection or geotransform found on file')
        sys.exit(1)
        
    dataset2 = Open(files, GA_Update)
    
    if geotransform is not None and geotransform != (0,1,0,0,0,1):
        dataset2.SetGeoTransform( geotransform )
    
    if projection is not None and projection != '':
        dataset2.SetProjection( projection )
    
    gcp_count = dataset.GetGCPCount()
    if gcp_count != 0:
        dataset2.SetGCPs(dataset.GetGCPs(), dataset.GetGCPProjection())
    
    dataset = None
    dataset2 = None




