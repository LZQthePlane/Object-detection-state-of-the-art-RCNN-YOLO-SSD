### MobileNets: Efficient (deep) neural networks
When building object detection networks we normally use an existing network architecture, such as VGG or ResNet, and then use it inside the object detection pipeline. The problem is that these network architectures can be very large in the order of 200-500MB.
Network architectures such as these are unsuitable for resource constrained devices like smart phone due to their sheer size and resulting number of computations.

**MobileNets** differ from traditional CNNs through the usage of *depthwise separable convolution* ,The general idea behind depthwise separable convolution is to split convolution into two stages:   
 - *1. A 3×3 depthwise convolution.*   
 - *2. Followed by a 1×1 pointwise convolution.*   
 
The problem is that we sacrifice accuracy — MobileNets are normally not as accurate as their larger big brothers, but they are much more resource efficient.

