# Object-detection-based-on-MobileNetSSD
Project of object detection using MobileNet and SSD, apply in image video and webcam.   
(The code comments are descibed in chinese)

------
## ***Folder Intro***
### —image
Images/video to test the ability of the detector.

### —test_out
The result of my test images/videos.

### —openh264-XXX.dll
A *.dll* required to run on video-mode and to save it as *.mp4* if using **Windows** platform.

### —MobileNetSSD_deploy
Files save the pre-trained SSD-MobileNet caffe model.   
   - .prototxt file specifies the architecture of the neural network – how the different layers are arranged etc.
   - .caffemodel file stores the weights of the trained model.   
      
The MobileNet SSD was first trained on the COCO dataset (Common Objects in Context) and was then fine-tuned on PASCAL VOC reaching **72.7% mAP** (mean average precision). It was the art of the state, but now defeated by **YOLO v3** which reached both high speed and better accuracy.  
We can therefore detect 20 objects in images (+1 for the background class), including airplanes, bicycles, birds, boats, bottles, buses, cars, cats, chairs, cows, dining tables, dogs, horses, motorbikes, people, potted plants, sheep, sofas, trains, and tv monitors.

------
## ***Feature***
### Single Shot Detector
When it comes to deep learning-based object detection there are three primary object detection methods that you’ll likely encounter:  
*Faster R-CNNs  
You Only Look Once (YOLO)  
Single Shot Detectors (SSDs)*

**Faster R-CNNs** are likely the most “heard of” method for object detection using deep learning; however, the technique can be difficult to understand (especially for beginners in deep learning), hard to implement, and challenging to train.
Furthermore, even with the “faster” implementation R-CNNs (where the “R” stands for “Region Proposal”) the algorithm can be **quite slow, on the order of 7 FPS.

If we are looking for pure speed then we tend to use **YOLO** as this algorithm is much faster, capable of processing 40-90 FPS on a Titan X GPU. The super fast variant of YOLO can even get up to **155 FPS.** The problem with YOLO is that it leaves much accuracy to be desired.

**SSDs**, originally developed by Google, is a **balance between the two**. The algorithm is more straightforward than Faster R-CNNs.

### MobileNets: Efficient (deep) neural networks
When building object detection networks we normally use an existing network architecture, such as VGG or ResNet, and then use it inside the object detection pipeline. The problem is that these network architectures can be very large in the order of 200-500MB.
Network architectures such as these are unsuitable for resource constrained devices like smart phone due to their sheer size and resulting number of computations.

MobileNets differ from traditional CNNs through the usage of *depthwise separable convolution* ,The general idea behind depthwise separable convolution is to split convolution into two stages:   
*1. A 3×3 depthwise convolution.   
2. Followed by a 1×1 pointwise convolution.*  
The problem is that we sacrifice accuracy — MobileNets are normally not as accurate as their larger big brothers, but they are much more resource efficient.

------
## Result
### Image example
![my beautiful girlfriend](https://github.com/LZQthePlane/Object-detection-based-on-MobileNetSSD/blob/master/test_out/example_01.jpg)  
As you see,  it's not a 100% accurate algorithm for the detector **mistake the** ***table*** **in the left of image as a** ***sofa***.
