### Download the model files First
You can fork the *.ckpt* weights model [here](https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing), thanks to [gliese581gg](https://github.com/gliese581gg/YOLO_tensorflow).   
Then place it to `weights` file.    


### Usage Examples :
Put the test file (image only) to `test_images` file.   
 command line input: `python yolo.py` or `python yolo_tf.py`    


### Note
Two scripts here provided are nearly same, for `yolo_tf.py` uses some **tf constructor API** methods for functions like *Non-max Minimum*, etc. 
