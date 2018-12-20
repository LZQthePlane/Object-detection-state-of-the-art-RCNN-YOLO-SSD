### Download the model files First
Using the script file getModels.sh from command line.   
 - `sudo chmod a+x getModels.sh`
 - `./getModels.sh`   

This will download:   
- the yolov3.weights file (containing the pre-trained network’s weights)   
- the yolov3.cfg file (containing the network configuration)   
- the coco.names file which contains the 80 different class names used in the COCO dataset.

### Usage Examples :
Put the test file (iamge or video) under the same directory   
   
 - `python3 object-detection-based-on-YOLOv3.py --image=test.jpg`   
 - `python3 object-detection-based-on-YOLOv3.py --video=test.mp4`   
 - if no argument provided, it starts the webcam.


### Note
In this project, we use the *Darknet model* and load it by *opencv DNN module*. However, this function **do not support GPU** to speed up.   
So if you have a GPU and want to take the advantage of it, try **keras model** which can inter with *tensorflow* or *Theano* backend. You can get the *.h* model by converting the *.cfg & .weights* files, you can fork [here](https://github.com/qqwweee/keras-yolo3).
