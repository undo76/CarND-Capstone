This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

## ROS Nodes Description

### Waypoint Loader

This ROS python node is responsible of loading the waypoints from the `.csv` file and adding the desired speed to them. Then it publishes it to `/base_waypoints`. I haven't made any changes to this file.

### Waypoint Updater

This python node is responsible of publishing the next waypoints ahead to `/final_waypoints`. It is also responsible of decelerating to the stop line when there is a red traffic light. In order to do so it searches efficiently the closest point ahead of the car's current position using a `KDTree`. Then it publishes the next `LOOKAHEAD_WPS` (100 waypoints), modifiying its desired speed in case it needs to decelerate.

### Waypoint Follower

This node calculates the desired yaw and velocity of the car in order to follow the waypoints fed by `/final_waypoints`. This node is written in C++ and I haven't modified it.

### Twist Controller

This python node consists of two controllers, one for controlling the wheel, and other for controlling the throttle and the break pedals. I haven't modified the yaw controller. For implementing the velocity controller, I have used a low pass filter, to filter out the high frequency components of the signal and then I have fed it into a PID controller. The controlled variables are published to be consumed by the vehicle DBW.

### Traffic Light Detector

This python node subscribes to the car camera published to `/image_color`, and then, if the traffic light is red it publishes the number of the waypoint closest to the traffic light. Otherwise, it publishes `-1`.

The node has been implemented with Keras on Tensorflow, using a pre-trained VGG-16 convolutional neural network. See next section for details. 

```
@article{DBLP:journals/corr/SimonyanZ14a,
  author    = {Karen Simonyan and
               Andrew Zisserman},
  title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1409.1556},
  year      = {2014},
  url       = {http://arxiv.org/abs/1409.1556},
  archivePrefix = {arXiv},
}
```

## Traffic Ligth Classification

All the training process has been done in a Jupyter Notebook.

### Images capture

The first step before to train the traffic lights is capturing training and validation data from the simulator. This has been done using the next command.

```sh
rosrun image_view extract_images image:=/image_color 
```

Then I have classified the into directories following the next structure.

```
img_simulator
   ├── train
   │   ├── green
   │   ├── none
   │   ├── red
   │   └── yellow
   └── validation
       ├── green
       ├── none
       ├── red
       └── yellow
```

### Model

I have selected a pretrained VGG16 model as basis for my classifier. I have replaced the top fully connected layers for a single layer. I have also added a Global average pooling layer before connecting to the fully connected layers. 

```python
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

x = base_model.output
x = Dense(256, activation='relu')(x)
x = Dropout(.5)(x)
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False    

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'], )
```

As regularization I have applied an aggressive Dropout layer.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, None, None, 3)     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, None, None, 64)    1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, None, None, 64)    36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, None, None, 64)    0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, None, None, 128)   73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, None, None, 128)   147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, None, None, 128)   0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, None, None, 256)   295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, None, None, 256)   590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, None, None, 256)   590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, None, None, 256)   0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, None, None, 512)   1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, None, None, 512)   0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, None, None, 512)   0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               131328    
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 1028      
=================================================================
Total params: 14,847,044
Trainable params: 132,356
Non-trainable params: 14,714,688
_________________________________________________________________
```

### Training

As we don't have too many samples available, I have decided to augment the images with simple transformations. I have also freezed the VGG16 layers to prevent overfitting.

In 
```python
image_datagen_train = keras.preprocessing.image.ImageDataGenerator(zoom_range=0.1, 
                                                                   shear_range=0.1, 
                                                                   channel_shift_range=.05, 
                                                                   rotation_range=5,                                                                    
                                                                  )
image_generator_train = image_datagen_train.flow_from_directory(    
    './data/img_simulator/train/', 
    batch_size=16,
    target_size=(300, 400),
    shuffle=True)

image_datagen_val = keras.preprocessing.image.ImageDataGenerator()
image_generator_val = image_datagen_val.flow_from_directory(    
    './data/img_simulator/validation/', 
    batch_size=16,
    target_size=(300, 400),
    shuffle=True)

model.fit_generator(image_generator_train,
                    steps_per_epoch = 2600/16, 
                    validation_data = image_generator_val, 
                    validation_steps= 10,                                                            
                    epochs=10,
                    shuffle=True,
                    class_weight={0: 2, 1: 1, 2: 1, 3: 4}
                    )
```

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
