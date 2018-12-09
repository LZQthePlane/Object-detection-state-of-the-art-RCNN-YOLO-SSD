### Download the model files first
You can download the package by clicking [here](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz), then extract the needed `frozen_inference_graph.pb` file and put it in `model` directory.

### Usage Examples :
Put the test file (iamge or video) under the same directory   
   
`python3 object-det-seg-based-on-Mask-RCNN.py --image=test.jpg`   
`python3 object-det-seg-based-on-Mask-RCNN.py --video=test.mp4`   
if no argument provided, it starts the webcam.

### Note
In the `model` directory:
 - **mscoco_labels.names** contains all the objects for which the model was trained.   
 - **colors.txt** file containing all the colors used to mask objects of various classes.   
 - **frozen_inference_graph.pb** : The pre-trained weights.
 - **mask_rcnn_inception_v2_coco_2018_01_28.pbtxt** : The text graph file that has been tuned by the OpenCV’s DNN support group, so that the network can be loaded using OpenCV. (But DNN support only Intel GPU).
