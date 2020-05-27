# Project Write-Up

People Counter APP using OpenVINO toolkit.

## Explaining Custom Layers

The process behind converting custom layers involves the following. 
* Generate the Extension Template Files Using the Model Extension Generator
* Using Model Optimizer to Generate IR Files Containing the Custom Layer
* Edit the CPU Extension Template Files
* Execute the Model with the Custom Layer

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations:

The difference between model accuracy pre- and post-conversion was less since SSD Mobilenet V2 was unable to detect still persons in the video feed (therefore counted 7 persons instead of 6 irrespective of having 1 buffer frames).

The size of the model pre-conversion was 67Mb and post-conversion was 66Mb.

The inference time of the model post-conversion was 70ms and FPS was 14.
The pre-trained model (person-detection-retail-0013) was faster with inference time :47ms, FPS :21 and was more accurate.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are to detect number of persons in room (follow social distancing during pandemic) and have a proper queue system.

Each of these use cases would be useful because we are able to detect all the people in the frame and process the required information.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows
1. Lighting and dramatically reduce the accuracy of the model, therefore by giving false positive and false negatives.
2. Camera should be supported and image size should be resized to model input requirements.


## Model Research


In investigating potential people counter models, I tried this model:

- Model 1: [SSD MobileNet V2]
  - [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - Run the app for image using following commands:
  ```
  python main.py -i CAM -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```
  - Using  a video file
  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```
