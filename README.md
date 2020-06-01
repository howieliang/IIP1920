# Interactive Intelligent Products

**Responsible Lecturer: Dr. Rong-Hao Liang** 
*Assistant Professor, Future Everyday Group, Industrial Design, TU Eindhoven*
*Assistant Professor, Signal Processing Systems Group, Electrical Engineering, TU Eindhoven*
*Homepage: http://www.ronghaoliang.me/*

**Teaching Assistants:**
Dr. Zengrong Guo, *Post-doc Researcher, Future Everyday Group, Industrial Design, TU Eindhoven*
Ruben van Dijk, *PhD candidate, Future Everyday Group, Industrial Design, TU Eindhoven*

----------

**Sensors** not only enable the interactivity of products but also generate data. **Machine intelligence** leverages computational power and data to empower further the product to deal with design problems involved prediction, decision, and adaptation. This course aims to help the students understand the main paradigms in sensing, data collection, signal processing, and machine learning to apply them in meaningful design solutions of **intelligent interactive products**.

![An Interactive Intelligent Product]
(https://paper-attachments.dropbox.com/s_2872065C7B1453917D680BF73E65026F47C98466B9BC5AC6E5A3A3247AFA79E1_1590905704193_image.png)


The course combine theories and practices. Through a series of lectures and workshops, you will learn the principle and functionality of the sensors and machine intelligence. You will develop the ability to use these signal processing methods and machine learning algorithms to deal with everyday life problems where real-world complexity, uncertainty, and changing conditions make the use of these technical solutions a necessity. Understanding the type of problems that really will benefit from the utilization of signal processing methods and machine learning algorithms and creating realistic scenarios of use is equally important.

[**Arduino**](https://www.arduino.cc/)**,** [**Processing**](https://processing.org/), and [**Weka**](https://www.cs.waikato.ac.nz/~ml/weka/) are used as the development platform. Ready available example codes are available in [**Github repository**](https://github.com/howieliang/IIP1920), so you don’t have to implement the algorithms from scratch.


![Course Overview: Topics Covered in Interactive Intelligent Products](https://paper-attachments.dropbox.com/s_2872065C7B1453917D680BF73E65026F47C98466B9BC5AC6E5A3A3247AFA79E1_1590991337954_image.png)

# 0. Prerequisites
----------
## 0.1 Hardware and Equipments
- **Microcontrollers: Please get at least one of the following microcontroller boards from an online store.**
    - All the example codes used in this course were made for [Arduino Leonardo](https://en.wikipedia.org/wiki/Arduino). The small Arduino board that I used in the lecture is [DFRobot Beetle](https://www.kiwi-electronics.nl/beetle-arduino-leonardo?search=beetle), which is available in Kiwi electronics. [Teensy](https://www.pjrc.com/teensy/) and [Circuit Playground](https://www.adafruit.com/index.php?main_page=category&cPath=888)are also Arduino-compatible. For this course, I highly recommend [Circuit Playground](https://www.kiwi-electronics.nl/circuit-playground-developer-edition?search=Circuit%20playground&description=true) or [Circuit Playground Express](https://www.kiwi-electronics.nl/circuit-playground-express?search=Circuit%20playground&description=true) platforms, which have various on-board sensors so that you don't need to solder or interface additional sensors by soldering.
- **Sensors: Please get at least one of the following sensor modules from an online store.**
    - The 3-Axis Accelerometer module, which will be used throughout all the exercises. It's important to note that most of the accelerometers only support 3.3V power supply. For [Arduino](https://en.wikipedia.org/wiki/Arduino) users, you should either plug the sensor to the 3.3V or simply find a 5V-compatible [Adafruit ADXL335](https://www.kiwi-electronics.nl/ADXL335-5V-ready-triple-axis-accelerometer-3g-analog-out?search=Adafruit%205v%20adxl&description=true). I recommend this [LilyPad ADXL335](https://www.tinytronics.nl/shop/nl/arduino/lilypad/lilypad-adxl335-accelerometer) because it supports analog output, instead of an [I2C one](https://www.tinytronics.nl/shop/nl/sensoren/accelerometer-gyro/mpu-6050-accelerometer-en-gyroscope-3-axis-module-3.3v-5v) that needs more complex wiring and an extra library. 
    - for [Teensy](https://www.pjrc.com/teensy/) users, you have more choices because Teensy uses 3.3V by default. I recommend this [LilyPad ADXL335](https://www.tinytronics.nl/shop/nl/arduino/lilypad/lilypad-adxl335-accelerometer) because it supports analog output, instead of an [I2C one](https://www.tinytronics.nl/shop/nl/sensoren/accelerometer-gyro/mpu-6050-accelerometer-en-gyroscope-3-axis-module-3.3v-5v) that needs more complex wiring and an extra library.  If you would like to try more sensors, consider [Teensy Prop Shield](https://www.tinytronics.nl/shop/nl/teensy/accessoires/teensy-prop-shield-met-bewegingssensoren).
    - for [Circuit Playground](https://www.adafruit.com/index.php?main_page=category&cPath=888) / [Circuit Playground Express](https://www.kiwi-electronics.nl/circuit-playground-express?search=Circuit%20playground&description=true) users, you have an on-board accelerometer that works well, so you don't need to worry about this.
    - If you cannot get an accelerometer, I would recommend you to find other analog sensors (e.g., LDR, FSR, Capacitive Sensors) instead. The more reactive the sensor, the better, because it's easier for debugging.
- **Equipment (Optional)**
    - To connect the sensors and the microcontroller, you might need [solder equipment](https://www.tinytronics.nl/shop/index.php?route=product/search&search=solder), [wires, and breadboard](https://www.tinytronics.nl/shop/index.php?route=product/search&search=breadboard). For debugging, you might need a [multimeter](https://www.tinytronics.nl/shop/nl/prototyping/meten/budget-multimeter-dvm832). Please try to get your "home lab" ready so you will be free to explore. 
## 0.2 Processing and WEKA
We use [Processing](https://processing.org/) as the main software programming environment because it's powerful, simple, and versatile. It also support popular machine learning libraries such as [WEKA](https://www.cs.waikato.ac.nz/~ml/weka/). Please get the Processing IDE installed before lectures.


- **Installation**
    1. *Get the latest version* *(3+)* *from the Processing download page:* *https://processing.org/download/?processing* (Links to an external site.)
    2. Unzip the downloaded file and run the “processing” program. You should be able to run Processing without installation.
        - For Windows, it is recommended to move the program file to `D:\processing\` 
        - For Mac OS X, move it to `/Applications/` 
        - For GNU/Linux, move it to   `/usr/local/processing/` 
- **Overview**
    Every Processing project is a **sketch** **of codes**. There are multiple built-in examples that could help us understand sketch files.
![Run a Processing sketch](https://paper-attachments.dropbox.com/s_65B95333E870BFB3A45F21F2AE8608C0A4DDB2FF56F5DE17C39A0D709A1FADF8_1462427276091_processing-mouse2d-marked.png)

    As a Hello-World example for checking whether the installation is successful, please 
    1. Click **File > Examples…**
    2. Open **Basic > Input > Mouse2D** example.
    3. Hit the **Run** button to play with it.
    More tutorials on Processing can be found on the official “Hello Processing" website. http://hello.processing.org/
## 0.3 Arduino
We use [Arduino IDE](https://www.arduino.cc/) and the [Wiring](http://wiring.org.co/reference/) language for hardware configuration, regardless of which microcontrollers (e.g., [Arduino](https://en.wikipedia.org/wiki/Arduino), [Teensy](https://www.pjrc.com/teensy/), [Circuit Playground](https://www.adafruit.com/index.php?main_page=category&cPath=888)) you are using. Arduino IDE is an open-source micro-controller development platform that allows us to connect electronic devices, such as LED, servo motor, button, or sensor and build the features we want. Please get the Arduino IDE installed before lectures.
    
- **Install the Arduino Software**
    *Get the latest version from the Arduino download page:* [*https://www.arduino.cc/en/Main/Software*](https://www.arduino.cc/en/Main/Software). The page provides install files for Windows, Mac OS X, and Linux as shown below. Download the install file of your operating system. Open file after downloading and proceed with the installation. *(You could also take a look at the official documents for* [*Windows*](https://www.arduino.cc/en/Guide/Windows#toc2)*,* [*Mac*](https://www.arduino.cc/en/Guide/MacOSX#toc2)*, and* [*Linux*](https://www.arduino.cc/en/Guide/Linux#toc2)*.)* More tutorials of Arduino can be found [Here](https://www.arduino.cc/en/Tutorial/HomePage).


# 1. Introduction and Problem Formulation
----------
## 1.1 Introduction [slides](https://www.dropbox.com/s/4mczlvorezbzoh2/IIP1920_1_1.pdf?dl=0)
- **Overview**
- **Prerequisites**
- **Schedule**
- **Module 1: Machine Learning**
- **Module 2: Signal Processing**
- **Synergies Between The Two Modules**

Lecture Video: https://www.youtube.com/watch?v=a7Nz--vvmis&

## 1.2 Problem Formulation [slides](https://www.dropbox.com/s/tm9xhc70onqxr4a/IIP1920_1_2.pdf?dl=0)
- **Why Machine Learning?**
- **Formulate Supervised Learning problems**
- **Types of Learning**

Lecture Video (Part A): https://www.youtube.com/watch?v=PzeCHB3isQ0&

Lecture Video (Part B): https://www.youtube.com/watch?v=j_OeFWBzhAY&

# 2. Data Preparation and Serial Communication
----------
## 2.1 Data Preparation [slides](https://www.dropbox.com/s/tnwxw5plvzsm10n/IIP1920_2_1.pdf?dl=0)
- **CSV vs. ARFF**
- **Save data as ARFF files**
- **Load data as ARFF files**

Lecture Video: https://www.youtube.com/watch?v=j-jRzfZ6OK8&

## 2.2 Serial Communication [slides](https://www.dropbox.com/s/8sjtxuz3r2h0l2n/IIP1920_2_2.pdf?dl=0)
- **Timer and Uniform Sampling**
- **Send Sensor Data Stream to Processing**
- **Save Serial Data as ARFF files**
- **Bidirectional Serial Data Streaming between Arduino and Processing**

Lecture Video: https://www.youtube.com/watch?v=AFH0IgNL-dc&

# 3. Classification and Regression
----------
## 3.1 Classification [slides](https://www.dropbox.com/s/89vzrp62b3om5p3/IIP1920_3_1.pdf?dl=0)
- **Noise in Data**
- **Linear Support Vector Classification**
    - Linear Support Vector Classifier
    - Train a classifier from data
    - Confusion matrix and In-sample accuracy
    - Overfitting and out-of-sample accuracy
    - Regularization
    - k-fold Cross-Validation
    - Optimize a prediction model
    - Load a test set and get the out-of-sample accuracy
- **Real-Time Posture Classification**

Lecture Video: https://www.youtube.com/watch?v=e3dy9guTdso&

Lecture Video: https://www.youtube.com/watch?v=qF5q2biOkSY&

## 3.2 Regression [slides](https://www.dropbox.com/s/jryd36lx89iwq3i/IIP1920_3_2.pdf?dl=0)
- **Linear Regression**
    - Linear Regressor
    - Train a regressor from data
    - Loss function
    - Linear Regression in higher dimension
- **Real-Time Posture Regression**

Lecture Video: https://www.youtube.com/watch?v=5PYb6YdTido&

# 4. Time-Series Signal Processing
----------
## 4.1 Time-Series Signal Processing [slides](https://www.dropbox.com/s/yiysva470k1sp6b/IIP1920_4_1.pdf?dl=0)
- **Visualizing Sensor Data Stream**
- **Segmentation and Activation Threshold**
- **Windowing and Statistic Feature**

Lecture Video: https://www.youtube.com/watch?v=RJz1QHV5VXA&

## 4.2 ****Real-Time Motion Classification and Regression [slides](https://www.dropbox.com/s/iwoma1zpg1sbiap/IIP1920_4_2.pdf?dl=0)
- **Real-time motion classification**
- **Real-time motion regression**
- **Using More Features**

Lecture Video: https://www.youtube.com/watch?v=QL4f3nL_XYY&

# 5. Evaluation and Reporting
----------
## 5.1 More Algorithms [slides](https://www.dropbox.com/s/aa5w86vgocuzij1/IIP1920_5_1.pdf?dl=0)
- **More Classifiers**
    - K-Nearest Neighbors Classifier
    - Kernel Support Vector Classifier
- **More Regressors**
    - Linear Support Vector Regressor
    - Kernel Support Vector Regressor
- **Hyper-parameter Tuning**

Lecture Video: https://www.youtube.com/watch?v=7JCryCb9Hr8&

## 5.2 Evaluation and Reporting [slides](https://www.dropbox.com/s/8itkwydp3z0q5an/IIP1920_5_2.pdf?dl=0)
- **More Evaluation Metrics**
    - True Positive Rate
    - False Positive Rate and Specificity
    - Precision
    - F1 score
    - Area Under ROC
    - More than Two Classes
    - Reporting
- **Validation Methods**
    - Validate the product with users
    - Comparing different algorithms 
    - Reporting

Lecture Video: https://www.youtube.com/watch?v=wP9pFwypqis&

# 6. Frequency-Domain Signal Processing
----------
## 6.1 Extraction Features in Frequency Domain [slides](https://www.dropbox.com/s/q3ef8w50fugpn32/IIP1920_6_1.pdf?dl=0)
- **Time Series in Frequency Domain**
- **Sampling Theory**
- **Fast Fourier Transform**
- **Window Overlapping vs. Information Loss**
- **Segmentation, Windowing, and Filtering**

Lecture Video: https://www.youtube.com/watch?v=a5hrJEwioRA&

## 6.2 Recognizing Sounds and Vibrations [slides](https://www.dropbox.com/s/2a0l8lp0osc1jul/IIP1920_6_2.pdf?dl=0)
- **Real-time audio classification using a microphone**
- **Real-time vibration classification using an accelerometer**

Lecture Video: https://www.youtube.com/watch?v=hcO-6h3YF04&

# 7. Sensor Fusion and Feature Selection
----------
## 7.1 Sensor Fusion and Context Recognition [slides](https://www.dropbox.com/s/lo7w715xrwya7lg/IIP1920_7_1.pdf?dl=0)
- **Human Activity Recognition Systems**
- **Synchronizing Sensor Data Streams**
- **Context Recognition using Sensor Fusion**

Lecture Video: https://www.youtube.com/watch?v=zm8rExDR3v0&

## 7.2 Feature Selection and Dimensionality Reduction [slides](https://www.dropbox.com/s/pvfgy0r8fxy2cai/IIP1920_7_2.pdf?dl=0)
****- **Feature Selection**
- **Dimensionality Reduction**

Lecture Video: https://www.youtube.com/watch?v=Xhk7XRY056w&

# 8. Spatial-Domain Signal Processing
----------
## 8.1 Extracting Features from Images [slides](https://www.dropbox.com/s/y98s082n8rvdhez/IIP1920_8_1.pdf?dl=0)
- **Image Recognition and Computer Vision**
- **Bitmap Data Structure**
- **Statistic Features: Histograms**
- **Structural Features: Edges**
- **Simplification and Downsampling**
- **Segmentation methods**

Lecture Video: https://www.youtube.com/watch?v=-p49Zmpv2EI&

## 8.2 Camera-based Activity Recognition [slides](https://www.dropbox.com/s/2t5hecw8l33ostm/IIP1920_8_2.pdf?dl=0)
- **Haar Cascade Classifier**
- **Application: Real-time Face Detection**
- **Marker-based Object Tracking**
- **Infra-red Cameras**

Lecture Video: https://www.youtube.com/watch?v=YzT65Gakxig&

# 9. Neural Networks and Wrap Up
----------
## 9.1 Neural Networks [slides](https://www.dropbox.com/s/36strjtu6i3s7p1/IIP1920_9_1.pdf?dl=0)
****- **Artificial Neural Network**
- **Neuron and Activation Functions**
- **Forward and Backward Propagation**
- **Gradient Descent and Learning Rate**
- **Training an ANN**
- **Convolutional Neural Network**

Lecture Video (Part A): https://www.youtube.com/watch?v=oV_I8WlBCOE&

Lecture Video (Part B): https://www.youtube.com/watch?v=gwkk2lDYFaM&

## 9.2 Looking Back and Moving Forward [slides](https://www.dropbox.com/s/9czzt5xjoacw07l/IIP1920_9_2.pdf?dl=0)

Lecture Video: https://www.youtube.com/watch?v=QnVbIXzju_A&

----------
