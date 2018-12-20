### Download the model files First
In [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) you can download many tensorflow pretrained models implemented by different algorithms and networks. The script here in this repository is for COCO-trained models.   

Inside the un-tar'ed directory, you will find:   
* a graph proto (`graph.pbtxt`)
* a checkpoint
  (`model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta`)
* a frozen graph proto with weights baked into the graph as constants
  (`frozen_inference_graph.pb`) to be used for out of the box inference. **Therefore, put this file under the `models` directory.**

### Usage Examples :
Put the test file (iamge or video) under the same directory   
   
 - `python3 object_detection_tf_API.py --image=test.jpg`   
 - `python3 object_detection_tf_API.py --video=test.mp4`   
 - if no argument provided, it starts the webcam.


### Note
