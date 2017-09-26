import numpy as np
##############################################################
from urllib.request import urlretrieve
from os.path import isfile
from tqdm import tqdm
import sys
import csv
import pickle
import numpy as np
import math
import os
import ntpath
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.python.control_flow_ops = tf
from sklearn.utils import shuffle
import pandas as pd

log_files_dir = "data/"
total_samples = 0

##############################################################
##############################################################
DEBUG_ALL_OFF = True
if DEBUG_ALL_OFF == False:
    DEBUG_IMG_PREPROCESS_ON = True
    DEBUG_ON = True
    DEBUG_SHOW_IMG = True
else:
    #### DO NOT MODIFY ####
    DEBUG_IMG_PREPROCESS_ON = False
    DEBUG_ON = False
    DEBUG_SHOW_IMG = False
    
def my_print(*arg):
    if DEBUG_ON:
        print(arg)

# generator function
def dbg_data_augmented(dlog, batch_size):
    X_train = []
    y_train = []
    global total_samples
    num_samples = 0
    my_print("dbg_data_augmented: IN")

    for ix in range(batch_size):            
        X_data, y_data = data_load_augmented(dlog)
        
                   
##############################################################
##############################################################

img_height = 160 #shape[0]
img_width = 320 #shape[1]
img_channels = 3
img_crop_bottom = 30
img_crop_top = 30
img_crop_top_val = img_height - img_crop_top

if 1:
    vertices = np.array([[(0,img_height-img_crop_bottom),\
                          (0,img_height-img_crop_top_val),\
                          (img_width,img_height-img_crop_top_val),
                          (img_width,img_height-img_crop_bottom)]],
                        dtype=np.int32)

else:
    toplencol_offset = 100
    toplenrow_offset = 0
    vertices = np.array([[(0,img_height-img_crop_bottom),\
                          (0,img_height-img_crop_top_val),\
                          (img_width/2-toplencol_offset, \
                           img_height/3 + toplenrow_offset), \
                          (img_width/2+toplencol_offset, \
                           img_height/3 + toplenrow_offset), \
                          (img_width,img_height-img_crop_top_val),
                          (img_width,img_height-img_crop_bottom)]],
                        dtype=np.int32)
                
img_final_height = int(img_height/2)
img_final_width = int(img_width/2)
img_final_channels = img_channels

def img_show(img, title = "No Title", cmap=plt.rcParams['image.cmap']):
    if DEBUG_SHOW_IMG:
        #plt.interactive(True)
        #plt.ion()
        plt.title(title)
        plt.imshow(img, cmap=cmap)
        plt.show()
        #plt.pause(0.001)
        #time.sleep(0.5)
        
import collections
img_saved_num = 5
img_saved_weights =[[1.0,  0.0,  0.0, 0.0, 0.0], \
                    [0.2,  0.8,  0.0, 0.0, 0.0], \
                    [0.1,  0.3,  0.7, 0.0, 0.0], \
                    [0.05, 1.05, 0.2, 0.6, 0.0], \
                    [0.05, 0.05, 0.1, 0.2, 0.6]]
img_saved = collections.deque(maxlen = img_saved_num)

def img_saved_process(img):
    global img_saved
    img_saved.append(img)
    num_imgs = len(img_saved)
    my_print("Weights used {}.{}".format(num_imgs, img_saved_weights[num_imgs-1]))
    for ix, img in enumerate(img_saved):
        img = img + img*img_saved_weights[num_imgs-1][ix]

    return img
    
# Normalize between -0.5 and 0.5
def img_normalize1(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

# Normalize between -1 and 1
def img_normalize2(img):
    my_print("img_normalize2: IN")
    return (img - 128.0)/128.0

def img_region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    global vertices

    #defining a blank mask to start with
    mask = np.zeros_like(img)
    #my_print("Image shape {}".format(img.shape))
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

def img_crop(img):
    shape = img.shape
    my_print("img_crop: IN {}".format(shape))
    img = img[math.floor(shape[0] - img_crop_top_val):\
              shape[0]-img_crop_bottom, 0:shape[1]]
    my_print("img_crop: OUT")
    return img

def img_preprocess(img):

    #img = img_region_of_interest(img)
    #img_show(img, title="Region of interest")
    
    img = img_crop(img)
    img_show(img, title="Cropped image")
    
    img = img_normalize2(img)
    img_show(img, title="Normalized image")

    #img = img_saved_process(img)
    #img_show(img, title="Normalized image", cmap='gray')
    img = cv2.resize(img, (img_final_width, img_final_height), cv2.INTER_AREA)
    img_show(img, title="Resized image")
    
    return img

def img_flip(img):
    
    #flip image along the X-axis
    img = cv2.flip(img, 1)
    img_show(img, title="Flipped image")
    
    return img

def img_brighten(img, factor):

    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    ########## Code from Vivek Yadav's blog post on behavioral cloning
    random_bright = .25+factor

    img[:,:,2] = img[:,:,2]*random_bright

    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    ###########
    img_show(img, title="Brighten/Darken")

    return img

def img_shift(img, steer_angle, trans_range, w_shift, h_shift):
    ########## Code from Vivek Yadav's blog post on behavioral cloning
    tr_x = trans_range*w_shift-trans_range/2
    steer_angle = steer_angle + tr_x/trans_range*2*.2
    tr_y = 40*h_shift-40/2
    
    transform_matrix = np.float32([[1,0,tr_x],[0,1,tr_y]])
    img = cv2.warpAffine(img, transform_matrix, (img_width,img_height))
    ###########
    img_show(img, title="Shifted image")
    
    return img, steer_angle

def img_preprocess_all(X_train):
    X_transformed = []
    my_print("img_preprocess_all: IN")
    for img in X_train:
        
        img = img_preprocess(img)
        
        X_transformed.append(img)

    return X_transformed

def img_augment_all(X_train, y_train):
    X_transformed = []
    y_transformed = []
    for img, y in zip(X_train, y_train):
        
        img = img_flip(img)

        y = y*(-1)
        
        X_transformed.append(img)
        y_transformed.append(y)

    return X_transformed, y_transformed

##############################################################
###### Load Data
##############################################################

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def adjust_angle_left(angle):
    angle += 0.08
    return angle

def adjust_angle_right(angle):
    angle -= 0.08
    return angle


def data_load_log_files():
    global log_files_dir
    log_files = os.listdir(log_files_dir)
    dlog = []
    
    for log_file in log_files:
        if os.path.isdir(log_file):
            print("{} is a directory. not a log file".format(log_file))
            continue
        
        if log_file.find("driving_log") == -1:
            print("{} is not a log file".format(log_file))
            continue
        
        with open(log_files_dir+log_file, 'rt') as f:
            print("**** Reading {}".format(log_file))
            reader = csv.reader(f)
            for line in reader:
                dlog.append(line)

    my_print("dlog: {} lines".format(len(dlog)))
    return dlog

def img_aug_probs(num_imgs):

    parms = {}

    parms['img'] = np.random.randint(1, num_imgs)
    
    parms['lrc'] = np.random.randint(3)

    parms['bright'] = np.random.uniform()
    
    parms['flip'] = np.random.randint(2)

    parms['w_shift'] = np.random.uniform()

    parms['h_shift'] = np.random.uniform()
    
    my_print("image_augs {}".format(parms))
    return parms

def data_load_augmented(dlog):

    while True:
        parms = img_aug_probs(len(dlog))
    
        # Line from dlog
        line = dlog[parms['img']]
        
        my_print("data_load_augmented: IN ")
        #Check if steering angle is valid
        try:
            steer_angle = float(line[3])
        except ValueError:
            print("Not a float {}".format(line[3]))
            continue

#        if(steer_angle == 0):
#            continue
        
        lrc = parms['lrc']
        #Check which image to pick
        if(lrc == 0):
            img_file = line[0]
            lrc_str = "center"
        elif(lrc == 1):
            img_file = line[1]
            steer_angle = adjust_angle_left(steer_angle)
            lrc_str = "left"
        elif(lrc == 2):
            img_file = line[2]
            steer_angle = adjust_angle_right(steer_angle)
            lrc_str = "right"
            
        img_file = log_files_dir+'IMG/'+path_leaf(img_file)
    
        #Check if the image exists
        if(isfile(img_file) == False):
            my_print("Image file not found {}".format(img_file))
            continue
        else:
            img = mpimg.imread(img_file)
        
        img_show(img, title="Orig-"+lrc_str+":"+str(steer_angle))

        #Brighten ligten image
        img = img_brighten(img, parms['bright'])

        #Flip image
        if(parms['flip'] == 1):
            img = img_flip(img)
            steer_angle = -steer_angle

        img, steer_angle = img_shift(img, steer_angle, 100, \
                                     parms['w_shift'], parms['h_shift'])
        
        #Preprocess image - region_of_interst and Normalize
        img = img_preprocess(img)

        return img, steer_angle
        
def data_generator_augmented(dlog, batch_size):
    X_train = np.zeros((batch_size, \
                        img_final_height, \
                        img_final_width, \
                        img_final_channels))
    y_train = np.zeros(batch_size)
    global total_samples
    num_samples = 0
    
    while True:
        # Build the batch to be returned from dlog.
        # Each entry is one image only.

        for ix in range(batch_size):
            # Probabilistically pick the image and the augmentations
            # to be applied to image. Inspired by several online posts
            my_print("data_generator_augmented: ix {}. ".format(ix))

            X_data, y_data = data_load_augmented(dlog)

            X_train[ix], y_train[ix] = X_data, y_data
            
        total_samples += batch_size
        yield X_train, y_train

def data_load_gen(line, this_pass):

    my_print("data_load_gen: IN ")
    #Check if steering angle is valid
    try:
        steer_angle = float(line[3])
    except ValueError:
        my_print("Not a float {}".format(line[3]))
        return False, None, None
    
    #if(steer_angle == 0):
    #   continue

    #Check which image to pick
    if(this_pass == "center"):
        img_file = line[0]
        lrc_str = "center"
    elif(this_pass == "left"):
        img_file = line[1]
        steer_angle = adjust_angle_left(steer_angle)
        lrc_str = "left"
    elif(this_pass == "right"):
        img_file = line[2]
        steer_angle = adjust_angle_right(steer_angle)
        lrc_str = "right"
    elif(this_pass == "flip"):
        img_file = line[0]
        lrc_str = "center"
        
    img_file = log_files_dir+'IMG/'+path_leaf(img_file)
    
    #Check if the image exists
    if(isfile(img_file) == False):
        my_print("Image file not found {}".format(img_file))
        return False, None, None
    else:
        img = mpimg.imread(img_file)
        
    img_show(img, title="Orig-"+lrc_str+":"+str(steer_angle))
    
    #Flip image
    if(this_pass == "flip"):
        img = img_flip(img)
        steer_angle = -steer_angle
        
    #Preprocess image - region_of_interst and Normalize
    img = img_preprocess(img)
        
    return True, img, steer_angle
        
def data_generator_gen(dlog, batch_size, this_pass):
    X_train = np.zeros((batch_size, \
                        img_final_height, \
                        img_final_width, \
                        img_final_channels))
    y_train = np.zeros(batch_size)
    global total_samples
    ix = 0
    while True:
        # Build the batch to be returned from dlog.
        # Each entry is one image only.

        for line in dlog:
            # Probabilistically pick the image and the augmentations
            # to be applied to image. Inspired by several online posts
            my_print("data_generator_gen: ix {}. ".format(ix))

            ret, X_data, y_data = data_load_gen(line, this_pass)

            if(ret == False):
                continue
            
            X_train[ix], y_train[ix] = X_data, y_data

            ix += 1
            if(ix >= batch_size-1):
                total_samples += ix+1
                yield X_train, y_train
                ix = 0
                    
##############################################################
##############################################################
#### Build Keras model
##############################################################
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import ELU, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback

RANDOM_INIT_FUNC = 'uniform' 

def model_baseline():
    model = Sequential()
    # Layer 1
    #model.add(BatchNormalization(axis=1))
    model.add(Convolution2D(3, 1, 1, init=RANDOM_INIT_FUNC, \
                            input_shape=(img_final_height, \
                                         img_final_width, \
                                         img_final_channels)))
    
              
    model.add(Convolution2D(24, 3, 3, init=RANDOM_INIT_FUNC))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))    
    
    #Layer 2
    model.add(Convolution2D(36, 3, 3, init=RANDOM_INIT_FUNC))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    #Layer 3
    model.add(Convolution2D(48, 3, 3, init=RANDOM_INIT_FUNC))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    #Layer 4
    model.add(Convolution2D(64, 3, 3, init=RANDOM_INIT_FUNC))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    #Flatten
    model.add(Flatten())

    # ** Dropouts after each FC critical!! 
    # FC Layer 1
    model.add(Dense(1024, init=RANDOM_INIT_FUNC))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    # FC Layer 2
    model.add(Dense(512, init=RANDOM_INIT_FUNC))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    # FC Layer 3
    model.add(Dense(64, init=RANDOM_INIT_FUNC))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    # FC Layer 4
    model.add(Dense(16, init=RANDOM_INIT_FUNC))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    
    # Compile and train the model
    model.compile('adam', 'mse')

    return model

##############################################################
##############################################################
### Save the model
def model_save(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    my_print("Saved model to disk")

class model_call_back(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        global total_samples
        print("Epoch begin...")
        total_samples = 0
        
    def on_epoch_end(self, epoch, logs={}):
        global total_samples
        print("Epoch End. Total number of samples in batch = {}".\
              format(total_samples))
        total_samples = 0
        
##############################################################        
##############################################################
if __name__ == "__main__":

    #Collect all logs in an array
    dlog = data_load_log_files()

    print("** Number of samples {}".format(len(dlog)))
    #TODO: Split the dataset for validation 20% of the dlog
    
    if DEBUG_IMG_PREPROCESS_ON:
        #Image preprocessing
        X_train, y_train = dbg_data_augmented(dlog, 10)
        
        my_print("DBG: Size of data set = {}".format(len(X_train)))
        
        sys.exit("DBG: IMG preprocessing debug")
        
    # Create keras model    
    model = model_baseline()

    epoch_events = model_call_back()

    passes = ["center", "left", "right", "flip"]
    for ix, this_pass in enumerate(passes):
        if(ix != 0):
            #Load previous model
            print("************* Pass {}: {}. Loading Weights ***************".\
                  format(this_pass, ix+1))
            model.load_weights("model.h5")
        else:
            print("************* Pass {}: {}. No weights to load ************".\
                  format(this_pass, ix+1))
            
        model.fit_generator(
            data_generator_gen(dlog, 64, this_pass),
            samples_per_epoch=6400,
            nb_epoch=6,
            validation_data=None,
            nb_val_samples=None,
            callbacks=[epoch_events])
        model_save(model)

    ### Next pass is to augment data and train
    print("************* Final Pass {}: {}. Image augmentation ***************".\
          format(this_pass, ix+1))
    # Load Weights
    model.load_weights("model.h5")
    
    #Train
    model.fit_generator(
        data_generator_augmented(dlog, 64),
        samples_per_epoch=19200,
        nb_epoch=9,
        validation_data=None,
        nb_val_samples=None,
        callbacks=[epoch_events])
    
    #Save the final model
    model_save(model)

##############################################################
##############################################################
##############################################################
##############################################################
