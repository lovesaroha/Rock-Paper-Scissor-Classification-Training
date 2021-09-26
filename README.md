# Rock-Paper-Scissor-Classification-Training
Training a keras model to recognize real images of rock, paper, scissors (hand) in order to classify an incoming image. <br/>
Download training images from [laurencemoroney-blog.appspot.com/rps.zip](https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip).<br/> 
Download validation images from [laurencemoroney-blog.appspot.com/rps-test-set.zip](https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip). <br/>
Download inception model weights for transfer learning from [mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5](https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5)

## Requirements
- Python 3 or higher.

## Packages

### Numpy
```bash
py -m pip install numpy
```

### Tensorflow
```bash
py -m pip install tensorflow
```

## Start Training
```bash
py convolutional_neural_network.py
```
```bash
py transfer_learning.py
```







