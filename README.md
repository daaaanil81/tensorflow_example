## tensorflow_example

Sample example how to load a Tensorflow Object detection API v2 model and serve prediction in C++  

## Download models

1. Object detection

```bash
https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
```

2. Animels detection

```bash
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz
```

## Build and installation

1. Tensorflow 2.8.0
2. CUDA 11.2
3. cuDNN 8.1
4. Bazel 4.2.1
5. Protobuf 3.9.2
6. OpenCV 4.3.0

### Build Docker image

```bash
docker build . -t daaaanil81/tensorflow_example
```

### Compile source

1. Extract model from archive model/saved_model

```bash
tar -xf model/saved_model.tar.gz -C ./model/
```
2.  Start container and mount the model volume

```bash
docker run -it --name "tensorflow" --detach-keys="ctrl-x" -v /Users/danilpetrov/Documents/tensorflow_example/:/root/tensorflow_example daaaanil81/detection /bin/bash
```
3. directory structure

```
-|/model/
    	|--saved_model
            |--assets/
            |--saved_model.pb
            |-- ...
```

4. Build the project using cmake

```bash
root@8122f3e1dc5b:/root/tensorflow_example# mkdir build
root@8122f3e1dc5b:/root/tensorflow_example# cd build && cmake ..
root@8122f3e1dc5b:/root/tensorflow_example# make
```
