**<u>Behavioral Cloning Project</u>**

**<u>Mandar Joshi, November Cohort</u>**

## Summary

The video below shows the results of cloning driving behavior solely
based on learning through camera data from 3 cameras (left, center and
right) in front of the car in the simulated environment. What follows is
a description of the data, data augmentation, neural network
description.

## 

Link to video: <https://www.youtube.com/watch?v=75uIQI34ikI>

## Model Architecture

The model is an implementation in Keras based on Nvidia’s CNN model with
some variations in the fully connected layer and the input layers.

Shown below is the diagram of the training system.

<img src="./media/image1.png" style="width:6.5in;height:3.08403in"
alt="A diagram of a machine AI-generated content may be incorrect." />

Shown below is the architecture of the Neural Network

<img src="./media/image2.png" style="width:4.33333in;height:6.18056in"
alt="A diagram of a diagram of a variety of layers AI-generated content may be incorrect." />

Find a detailed description of the Nvidia model –

<https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/>

## Data

Training data (provided by Udacity) consisted of individual frames
(~8000) from a video of a car being driven in a simulated environment
with each frame being annotated with the cars steering angle, throttle
and speed at that instant.

The same data was used to create augmented data for training to ensure
that autonomous driving works under different conditions of light, road,
environment, etc. However, before augmentation the images were
preprocessed through normalization, cropping for region-of-interest,
resize.

### Data Preprocessing

- Image normalization: Image has been normalized in the range -1 to 1

- Region of interest/cropping: the top and bottom 30 pixels are removed
  from the image. The top does not contribute in any meaningful way to
  the driving model and the bottom part shows the hood of the car in
  different locations based on if the image is from the
  center/left/right camera. This must be removed so that it does not
  factor in the learning when using image augmentation (which is used to
  train to keep the car within the lanes),

- Resize image: The images were used in their original size of 160x320x3
  for all pre-processing/processing and eventually resized by a factor
  of 2 in terms of the height and width (80x160x3). This does not seem
  to have made any difference to the training quality.

### 

### Data Augmentation

Each augmented image was a result of the following augmentations applied
to an image based on generated probabilities in img_aug_probs(). These
generated probabilities would determine the type of camera image to pick
(left, right, center) and the type of augmentation applied. At the end
you would get an augmented image.

- Brighten/Darken: The image would be brightened or darkened based on
  randomly generated value from a normal distribution. (See below for
  examples)

- Flipped: The image would be flipped or not depending on a randomly
  generated integer 0 or 1. The steering angle would be adjusted
  accordingly. (See below for examples)

- Shift: The image would be shifted based on the values generated for
  width shift and height shift. The steering angle would be adjusted
  accordingly. (Red. Inspired by Vivek Yadav blog on behavioral learning
  in the Dec cohort) (See below for examples)

## Training

### Phases

- The training was divided into two main parts. The idea was to train
  the first phase using given data and <u>fine-tuning</u> the model in
  phase 2 by generating and training on augmented data.

  - Phase 1: In the first phase the data was largely used unaltered to
    train the model. The only data that was generated was to use flipped
    images (and steering angles) to eliminate the skew in the data that
    results because of more left turns than right turns on the track.
    (data_generator_gen() & load_data_gen())

  - Phase 2: In this phase the model was trained using augmented data.
    The details of the data augmentation performed are listed above.
    (data_generator_augmented(), load_data_augmented())

- Fit_generator, Epochs, batch_size, etc.

  - Keras fit_generator (that makes use of python generators) was
    required to ensure that there is no need to load the entire data in
    memory which will result in running out of memory.

  - A batch_size of 64 was used

  - Phase 1 used 6 epochs each to train the model with center, left,
    right and flipped data. 6400 samples per epoch.

  - Phase 2 generated augmented data on the fly based on generated
    probabilities to select the type and extent of augmentation (ref:
    img_aug_probs())

Example 1:

<img src="./media/image3.png" style="width:6.49097in;height:3.61111in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.10.47%20PM." /><img src="./media/image4.png" style="width:6.5in;height:3.60208in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.11.03%20PM." /><img src="./media/image5.png" style="width:6.49097in;height:3.54653in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.11.15%20PM." /><img src="./media/image6.png" style="width:6.5in;height:2.43542in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.11.25%20PM." /><img src="./media/image7.png" style="width:6.5in;height:2.31458in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.11.34%20PM." /><img src="./media/image8.png" style="width:6.5in;height:3.52778in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.12.05%20PM." />

Example2:

<img src="./media/image9.png" style="width:6.5in;height:3.56458in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.13.00%20PM." /><img src="./media/image10.png" style="width:6.5in;height:3.54653in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.13.09%20PM." /><img src="./media/image11.png" style="width:6.5in;height:3.62986in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.13.17%20PM." /><img src="./media/image12.png" style="width:6.49097in;height:3.50903in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.13.27%20PM." /><img src="./media/image13.png" style="width:6.49097in;height:2.5in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.13.35%20PM." /><img src="./media/image14.png" style="width:6.5in;height:2.42569in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.13.43%20PM." /><img src="./media/image15.png" style="width:6.49097in;height:3.49097in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.13.51%20PM." />

Example 3:

<img src="./media/image16.png" style="width:6.5in;height:3.62986in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.14.03%20PM." /><img src="./media/image17.png" style="width:6.49097in;height:3.56458in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.14.12%20PM." /><img src="./media/image18.png" style="width:6.49097in;height:3.50903in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.14.19%20PM." /><img src="./media/image19.png" style="width:6.5in;height:2.47222in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.14.26%20PM." /><img src="./media/image20.png" style="width:6.5in;height:2.50903in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.14.33%20PM." /><img src="./media/image21.png" style="width:6.49097in;height:3.50903in"
alt="../../../../Desktop/Screen%20Shot%202017-02-04%20at%205.14.42%20PM." />
