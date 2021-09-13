# Car Recognition

Yunus Emre Şencan

## Introduction

Smart cities are currently on the rise all around the world. One crucial part of these cities are is smart city surveillance systems, which monitors the city to identify the incidents that could require intervention from the relevant personnel. These systems consist of mobile or fixed cameras around the city, and incident management units (IMUs) on the ground to evaluate the image coming from cameras using machine learning systems. 

One challenge that these systems face speed limitations of the wireless file transfers makes it impossible to constantly send high-resolution video feed, which are necessary to identify vehicles that got in the accident, to IMUs. To solve this issue, we could seperate the video feed into two parts: the important parts that contains images in which accidents could have happened, and the unimportant parts which are any other part like regular flow of the city.

The purpose of this repository is to create a solution to find the possibility of a car being in a video frame. This probability will further be used to assign a Score of Interestingness (SI) value to decide which parts are important.

## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

We use the Cars Dataset from Stanford University, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/random.jpg)

You can get it from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

```bash
$ cd Car-Recognition
$ wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
$ wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```

## ImageNet Pretrained Models

Download [ResNet-152](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing) and [ResNet-50](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5) into models folder.

## Usage

### Data Pre-processing
This script extracts 8,144 training images, and split them by 80:20 rule (6,515 for training, 1,629 for validation) to feed to training:
```bash
$ python pre_process.py
```

### Train
The training scripts train the 50-layered ResNet-50 model as default. To change that to 152 layered ResNet-152, change line 22:
```bash
model = resnet152_model(img_height, img_width, num_channels, num_classes)
```

To train the model:
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/train.jpg)

### Analysis
Update "model_weights_path" in "utils.py" with your best model, and use 1,629 validation images for result analysis:
```bash
$ python analyze.py
```

#### Validation acc:
**97.54%**

#### Confusion matrix:

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/confusion_matrix.jpg)

### Test
```bash
$ python test.py
```

#### Test acc:
**88.88%**


### Demo
To select 20 random images from test dataset and run the model on them:

```bash
$ python demo.py
```

```bash
$ python demo.py
class_name: Lamborghini Reventon Coupe 2008
prob: 0.9999994
```

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/0_out.png)  | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/1_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/2_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/3_out.png) |
|Hyundai Azera Sedan 2012, prob: 0.99|Hyundai Genesis Sedan 2012, prob: 0.9995|Cadillac Escalade EXT Crew Cab 2007, prob: 1.0|Lamborghini Gallardo LP 570-4 Superleggera 2012, prob: 1.0|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/4_out.png)  | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/5_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/6_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/7_out.png) |
|BMW 1 Series Coupe 2012, prob: 0.9948|Suzuki Aerio Sedan 2007, prob: 0.9982|Ford Mustang Convertible 2007, prob: 1.0|BMW 1 Series Convertible 2012, prob: 1.0|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/8_out.png)  | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/9_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/10_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/11_out.png)|
|Mitsubishi Lancer Sedan 2012, prob: 0.4401|Cadillac CTS-V Sedan 2012, prob: 0.9801|Chevrolet Traverse SUV 2012, prob: 0.9999|Bentley Continental GT Coupe 2012, prob: 0.9953|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/12_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/13_out.png)| ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/14_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/15_out.png)|
|Nissan Juke Hatchback 2012, prob: 0.9935|Chevrolet TrailBlazer SS 2009, prob: 0.987|Hyundai Accent Sedan 2012, prob: 0.9826|Ford Fiesta Sedan 2012, prob: 0.6502|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/16_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/17_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/18_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/19_out.png)|
|Acura TL Sedan 2012, prob: 0.9999|Aston Martin V8 Vantage Coupe 2012, prob: 0.5487|Infiniti G Coupe IPL 2012, prob: 0.2621|Ford F-150 Regular Cab 2012, prob: 0.9995|
