# Object-detection-state-of-the-art
Application of object detection methods state-of-the-art, including YOLOv3, SSD up to now. 
(The code comments are partly descibed in chinese)

------
When it comes to deep learning-based object detection there are three primary object detection methods that you’ll likely encounter:  
*Faster R-CNNs  
You Only Look Once (YOLO)  
Single Shot Detectors (SSDs)*

**Faster R-CNNs** are likely the most “heard of” method for object detection using deep learning; however, the technique can be difficult to understand (especially for beginners in deep learning), hard to implement, and challenging to train.
Furthermore, even with the “faster” implementation R-CNNs (where the “R” stands for “Region Proposal”) the algorithm can be **quite slow, on the order of 7 FPS.

If we are looking for pure speed then we tend to use **YOLO** as this algorithm is much faster, capable of processing 40-90 FPS on a Titan X GPU. The super fast variant of YOLO can even get up to **155 FPS.** The problem with YOLO is that it leaves much accuracy to be desired.

**SSDs**, originally developed by Google, is a **balance between the two**. The algorithm is more straightforward than Faster R-CNNs.

------
## Results and Comparision

<figure class="birds">
   <img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/MobileNet-SSD/test_out/bird_out.jpg" width="500">
   <img src="https://github.com/LZQthePlane/Object-detection-state-of-the-art/blob/master/YOLOv3/test_out/bird_out.jpg"  width="500">
</figure>
