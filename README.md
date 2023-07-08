# yolov3(Darknet) -> ONNX -> TensorRT

This repo converting **yolov3 and yolov3-tiny** darknet model to TensorRT model.

If you want to convert **yolov3 or yolov3-tiny** pytorch model, need to convert model from pytorch to DarkNet. Check my [yolov3-pytorch repo](https://github.com/2damin/yolov3-pytorch)

## ENV INFO

**ONNX == 1.9.0**

**TensorRT == 8.2.1.8**

- CUDA == 10.2

- OpenCV == 4.5.5.62


## Prerequisite

### install packages
```bash
pip install -r requirements.txt
```

## Sample

### 1. download darknet weights

YOLOv3-tiny.weights : https://pjreddie.com/media/files/yolov3-tiny.weights  


```bash
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

### 2. weights > onnx


```bash
#converting darkenet weights to onnx weights
python3 yolov3_to_onnx.py --cfg ${CFG_PATH} --weigths ${ONNX_PATH} --num_class ${num_of_classes}
python3 yolov3_to_onnx.py --cfg yolov3-tiny.cfg --weights yolov3-tiny.weights --num_class 80
```
```bash
uild_onnx_graph
layer loader
layer_type :  net 000_net
Layer of type yolo not supported, skipping ONNX node generation.
Layer of type yolo not supported, skipping ONNX node generation.
weight loader
001_convolutional
003_convolutional
005_convolutional
007_convolutional
009_convolutional
011_convolutional
013_convolutional
014_convolutional
015_convolutional
016_convolutional
019_convolutional
020_upsample
022_convolutional
023_convolutional
make_graph
make_model
graph YOLOv3-608 (
  %000_net[FLOAT, 1x3x416x416]
) optional inputs with matching initializers (
  %001_convolutional_bn_scale[FLOAT, 16]
  %001_convolutional_bn_bias[FLOAT, 16]
  %001_convolutional_bn_mean[FLOAT, 16]
  %001_convolutional_bn_var[FLOAT, 16]
  %001_convolutional_conv_weights[FLOAT, 16x3x3x3]
  %003_convolutional_bn_scale[FLOAT, 32]
  %003_convolutional_bn_bias[FLOAT, 32]
  %003_convolutional_bn_mean[FLOAT, 32]
  %003_convolutional_bn_var[FLOAT, 32]
  %003_convolutional_conv_weights[FLOAT, 32x16x3x3]
  %005_convolutional_bn_scale[FLOAT, 64]
  %005_convolutional_bn_bias[FLOAT, 64]
  %005_convolutional_bn_mean[FLOAT, 64]
  %005_convolutional_bn_var[FLOAT, 64]
  %005_convolutional_conv_weights[FLOAT, 64x32x3x3]
  %007_convolutional_bn_scale[FLOAT, 128]
  %007_convolutional_bn_bias[FLOAT, 128]
  %007_convolutional_bn_mean[FLOAT, 128]
  %007_convolutional_bn_var[FLOAT, 128]
  %007_convolutional_conv_weights[FLOAT, 128x64x3x3]
  %009_convolutional_bn_scale[FLOAT, 256]
  %009_convolutional_bn_bias[FLOAT, 256]
  %009_convolutional_bn_mean[FLOAT, 256]
  %009_convolutional_bn_var[FLOAT, 256]
  %009_convolutional_conv_weights[FLOAT, 256x128x3x3]
  %011_convolutional_bn_scale[FLOAT, 512]
  %011_convolutional_bn_bias[FLOAT, 512]
  %011_convolutional_bn_mean[FLOAT, 512]
  %011_convolutional_bn_var[FLOAT, 512]
  %011_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %013_convolutional_bn_scale[FLOAT, 1024]
  %013_convolutional_bn_bias[FLOAT, 1024]
  %013_convolutional_bn_mean[FLOAT, 1024]
  %013_convolutional_bn_var[FLOAT, 1024]
  %013_convolutional_conv_weights[FLOAT, 1024x512x3x3]
  %014_convolutional_bn_scale[FLOAT, 256]
  %014_convolutional_bn_bias[FLOAT, 256]
  %014_convolutional_bn_mean[FLOAT, 256]
  %014_convolutional_bn_var[FLOAT, 256]
  %014_convolutional_conv_weights[FLOAT, 256x1024x1x1]
  %015_convolutional_bn_scale[FLOAT, 512]
  %015_convolutional_bn_bias[FLOAT, 512]
  %015_convolutional_bn_mean[FLOAT, 512]
  %015_convolutional_bn_var[FLOAT, 512]
  %015_convolutional_conv_weights[FLOAT, 512x256x3x3]
  %016_convolutional_conv_bias[FLOAT, 255]
  %016_convolutional_conv_weights[FLOAT, 255x512x1x1]
  %019_convolutional_bn_scale[FLOAT, 128]
  %019_convolutional_bn_bias[FLOAT, 128]
  %019_convolutional_bn_mean[FLOAT, 128]
  %019_convolutional_bn_var[FLOAT, 128]
  %019_convolutional_conv_weights[FLOAT, 128x256x1x1]
  %020_upsample_scale[FLOAT, 4]
  %020_upsample_roi[FLOAT, 4]
  %022_convolutional_bn_scale[FLOAT, 256]
  %022_convolutional_bn_bias[FLOAT, 256]
  %022_convolutional_bn_mean[FLOAT, 256]
  %022_convolutional_bn_var[FLOAT, 256]
  %022_convolutional_conv_weights[FLOAT, 256x384x3x3]
  %023_convolutional_conv_bias[FLOAT, 255]
  %023_convolutional_conv_weights[FLOAT, 255x256x1x1]
) {
  %001_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%000_net, %001_convolutional_conv_weights)
  %001_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%001_convolutional, %001_convolutional_bn_scale, %001_convolutional_bn_bias, %001_convolutional_bn_mean, %001_convolutional_bn_var)
  %001_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%001_convolutional_bn)
  %002_maxpool = MaxPool[auto_pad = 'SAME_UPPER', kernel_shape = [2, 2], strides = [2, 2]](%001_convolutional_lrelu)
  %003_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%002_maxpool, %003_convolutional_conv_weights)
  %003_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%003_convolutional, %003_convolutional_bn_scale, %003_convolutional_bn_bias, %003_convolutional_bn_mean, %003_convolutional_bn_var)
  %003_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%003_convolutional_bn)
  %004_maxpool = MaxPool[auto_pad = 'SAME_UPPER', kernel_shape = [2, 2], strides = [2, 2]](%003_convolutional_lrelu)
  %005_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%004_maxpool, %005_convolutional_conv_weights)
  %005_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%005_convolutional, %005_convolutional_bn_scale, %005_convolutional_bn_bias, %005_convolutional_bn_mean, %005_convolutional_bn_var)
  %005_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%005_convolutional_bn)
  %006_maxpool = MaxPool[auto_pad = 'SAME_UPPER', kernel_shape = [2, 2], strides = [2, 2]](%005_convolutional_lrelu)
  %007_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%006_maxpool, %007_convolutional_conv_weights)
  %007_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%007_convolutional, %007_convolutional_bn_scale, %007_convolutional_bn_bias, %007_convolutional_bn_mean, %007_convolutional_bn_var)
  %007_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%007_convolutional_bn)
  %008_maxpool = MaxPool[auto_pad = 'SAME_UPPER', kernel_shape = [2, 2], strides = [2, 2]](%007_convolutional_lrelu)
  %009_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%008_maxpool, %009_convolutional_conv_weights)
  %009_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%009_convolutional, %009_convolutional_bn_scale, %009_convolutional_bn_bias, %009_convolutional_bn_mean, %009_convolutional_bn_var)
  %009_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%009_convolutional_bn)
  %010_maxpool = MaxPool[auto_pad = 'SAME_UPPER', kernel_shape = [2, 2], strides = [2, 2]](%009_convolutional_lrelu)
  %011_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%010_maxpool, %011_convolutional_conv_weights)
  %011_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%011_convolutional, %011_convolutional_bn_scale, %011_convolutional_bn_bias, %011_convolutional_bn_mean, %011_convolutional_bn_var)
  %011_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%011_convolutional_bn)
  %012_maxpool = MaxPool[auto_pad = 'SAME_UPPER', kernel_shape = [2, 2], strides = [1, 1]](%011_convolutional_lrelu)
  %013_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%012_maxpool, %013_convolutional_conv_weights)
  %013_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%013_convolutional, %013_convolutional_bn_scale, %013_convolutional_bn_bias, %013_convolutional_bn_mean, %013_convolutional_bn_var)
  %013_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%013_convolutional_bn)
  %014_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%013_convolutional_lrelu, %014_convolutional_conv_weights)
  %014_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%014_convolutional, %014_convolutional_bn_scale, %014_convolutional_bn_bias, %014_convolutional_bn_mean, %014_convolutional_bn_var)
  %014_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%014_convolutional_bn)
  %015_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%014_convolutional_lrelu, %015_convolutional_conv_weights)
  %015_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%015_convolutional, %015_convolutional_bn_scale, %015_convolutional_bn_bias, %015_convolutional_bn_mean, %015_convolutional_bn_var)
  %015_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%015_convolutional_bn)
  %016_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%015_convolutional_lrelu, %016_convolutional_conv_weights, %016_convolutional_conv_bias)
  %019_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%014_convolutional_lrelu, %019_convolutional_conv_weights)
  %019_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%019_convolutional, %019_convolutional_bn_scale, %019_convolutional_bn_bias, %019_convolutional_bn_mean, %019_convolutional_bn_var)
  %019_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%019_convolutional_bn)
  %020_upsample = Resize[coordinate_transformation_mode = 'asymmetric', mode = 'nearest', nearest_mode = 'floor'](%019_convolutional_lrelu, %020_upsample_roi, %020_upsample_scale)
  %021_route = Concat[axis = 1](%020_upsample, %009_convolutional_lrelu)
  %022_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [3, 3], strides = [1, 1]](%021_route, %022_convolutional_conv_weights)
  %022_convolutional_bn = BatchNormalization[epsilon = 9.99999974737875e-06, momentum = 0.990000009536743](%022_convolutional, %022_convolutional_bn_scale, %022_convolutional_bn_bias, %022_convolutional_bn_mean, %022_convolutional_bn_var)
  %022_convolutional_lrelu = LeakyRelu[alpha = 0.100000001490116](%022_convolutional_bn)
  %023_convolutional = Conv[auto_pad = 'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]](%022_convolutional_lrelu, %023_convolutional_conv_weights, %023_convolutional_conv_bias)
  return %016_convolutional, %023_convolutional
}
check_model
save onnx
onnx_path yolov3-tiny.onnx
```

### 3. onnx > trt

```bash
#building trt model and run inference
python3 onnx_to_tensorrt.py --cfg ${CFG_PATH} --onnx ${ONNX_PATH} --num_class ${num_of_classes} --input_img &{test_img_path}
python3 onnx_to_tensorrt.py --cfg yolov3-tiny.cfg --onnx yolov3-tiny.onnx --num_class 80 --input_img dog.jpg
```
```bash
engine_path yolov3-tiny.trt
None
build engine
fp16 : False
int8 : True
Loading ONNX file from path yolov3-tiny.onnx...
Beginning ONNX file parsing
Completed parsing of ONNX file
Building an engine from file yolov3-tiny.onnx; this may take a while...
[07/08/2023-20:51:29] [TRT] [W] TensorRT was linked against cuBLAS/cuBLAS LT 10.2.3 but loaded cuBLAS/cuBLAS LT 10.2.2
[07/08/2023-20:51:43] [TRT] [W] TensorRT was linked against cuBLAS/cuBLAS LT 10.2.3 but loaded cuBLAS/cuBLAS LT 10.2.2
[07/08/2023-20:51:43] [TRT] [W] TensorRT was linked against cuBLAS/cuBLAS LT 10.2.3 but loaded cuBLAS/cuBLAS LT 10.2.2
Completed creating Engine
[07/08/2023-20:51:43] [TRT] [W] TensorRT was linked against cuBLAS/cuBLAS LT 10.2.3 but loaded cuBLAS/cuBLAS LT 10.2.2
allocating buffers
Running inference on image dog.jpg...
infer : 0.009855031967163086
latency : 0.0641634464263916
[[ 67.44063323 158.04626069 139.51884581 216.21053655]
 [252.73754901  59.42544106 118.87930397  64.6066815 ]
 [287.15216059  67.97671175  54.14565435  48.51732691]] [16  2  7] [0.82787692 0.74259118 0.55927797]
Saved image with bounding boxes of detected objects to predict.png.
```

