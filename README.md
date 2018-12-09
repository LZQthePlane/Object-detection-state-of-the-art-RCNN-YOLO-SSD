# Object-detection-state-of-the-art
Application of object detection methods state-of-the-art, including YOLOv3, SSD, Mask-RCNN up to now.    
(The code comments are partly descibed in chinese)

------
## Introduction
When it comes to deep learning-based object detection there are three primary object detection methods that you’ll likely encounter:  
*Faster R-CNNs  
You Only Look Once (YOLO)  
Single Shot Detectors (SSDs)*

- **Faster R-CNNs** are likely the most “heard of” method for object detection using deep learning; however, the technique can be difficult to understand (especially for beginners in deep learning), hard to implement, and challenging to train.
Furthermore, even with the “faster” implementation R-CNNs (where the “R” stands for “Region Proposal”) the algorithm can be **quite slow, on the order of 7 FPS. **  
**Mask-RCNN** is a new member of RCNN series, which can not only detect objects but also segment its shape, and it was trained on specific *MSCOCO* dataset.

 - If we are looking for pure speed then we tend to use **YOLO** as this algorithm is much faster, capable of processing 40-90 FPS on a Titan X GPU. The super fast variant of YOLO can even get up to **155 FPS.** The problem with YOLO is that it leaves much accuracy to be desired.

 - **SSDs**, originally developed by Google, is a **balance between the two**. The algorithm is more straightforward than Faster R-CNNs. Here we used is SSD based on MobileNet, which simplifies the computation and run much faster to satisfy real-time need but lower the accuracy pretty much meanwhile.

**The newest updated version —— YOLOv3, has achieved very comparable accuracy than SSD while running much faster.**

------
## Results and Comparision
### ***MoboileNet-SSD in left, YOLOv3 in right*** 
<div style="float:left;border:solid 5px 000;margin:2px;"><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/MobileNet-SSD/test_out/bird_out.jpg" width="360"/><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/YOLOv3/test_out/bird_out.jpg"  width="360"/></div>
<div style="float:left;border:solid 5px 000;margin:2px;"><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/MobileNet-SSD/test_out/dinner_out.jpg" width="360"/><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/YOLOv3/test_out/dinner_out.jpg"  width="360"/></div>
<div style="float:left;border:solid 5px 000;margin:2px;"><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/MobileNet-SSD/test_out/group_out.jpg" width="360"/><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/YOLOv3/test_out/group_out.jpg"  width="360"/></div>
<div style="float:left;border:solid 5px 000;margin:2px;"><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/MobileNet-SSD/test_out/lab_out.gif" width="360"/><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/YOLOv3/test_out/lab_out.gif"  width="360"/></div>
<div style="float:left;border:solid 5px 000;margin:2px;"><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/MobileNet-SSD/test_out/webcam_out.gif" width="360"/><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/YOLOv3/test_out/me.gif"  width="360"/></div>   
   
   
### ***Mask-RCNN below***
<div style="float:left;border:solid 5px 000;margin:2px;"><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/Mask-RCNN/test_out/dinner_out.jpg" width="360" height="240"/><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/Mask-RCNN/test_out/home_out.jpg"  width="360" height="240"/></div>
<div style="float:left;border:solid 5px 000;margin:2px;"><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/Mask-RCNN/test_out/worktable_out.jpg" width="360" height="240"/><img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/Mask-RCNN/test_out/dogs_out.jpg"  width="360" height="240"/></div>   

------
## Note
In each seperated directory, you can see more details in its README.md.
