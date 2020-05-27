"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

D1 = 0
D2 = 0
current_count = 0
tcount = 0
k = 10
def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.55,
                        help="Probability threshold for detections filtering"
                        "(0.55 by default)")
    return parser

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''  
    global D1, D2, current_count, k, tcount
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 1)
            
            if box[1] ==1:
                if k>=10:
                    D1 = time.time()
                    current_count += 1
                    tcount+=1
                    k = 0
                D2 = time.time()
                
    if np.all(result[0,0,:,2] < args.prob_threshold):
        k+=1
        current_count=0
    s = D2 - D1
    return frame, s


def connect_mqtt():
    # Connect to the MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    #client = mqtt.Client()
    #client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    # Initialise the class
    plugin = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    ### TODO: Load the model through `infer_network` ###
    plugin.load_model(model, args.device, args.cpu_extension)
    net_input_shape = plugin.get_input_shape()
    
    if args.input =='CAM':
        input_stream = 0
        single_image = False
    elif args.input[-4:] in [".jpg", ".bmp"]:
        single_image = True
        input_stream = args.input
    else:
        single_image=False
        input_stream = args.input
        assert os.path.isfile(input_stream)
        
        
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)
    if not cap.isOpened():
        log.error("Unable to open video source")

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    ### TODO: Handle the input stream ###
    out = cv2.VideoWriter('out1.mp4', 0x00000021, 30, (width,height))
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Read from the video capture ###

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        ### TODO: Start asynchronous inference for specified request ###
        t1 = time.time()
        plugin.exec_net(p_frame)
        ### TODO: Wait for the result ###
        if plugin.wait() == 0:
            ### TODO: Extract any desired stats from the results ###
            result = plugin.extract_output()
            t2 = time.time()
            ### TODO: Get the results of the inference request ###
            s1=t2-t1
            ### TODO: Calculate and send relevant information on ###
            frame, s = draw_boxes(frame, result, args, width, height)
            t3 = time.time() - t2
            txt = "current_count: %d" %current_count + " total_count: %d" %tcount
            cv2.putText(frame, txt, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255, 0, 0), 2)
            txt1 = "duration: %d" %s
            cv2.putText(frame, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255, 0, 0), 2)
            txt2 = "Inference time: {:.3f}ms".format(s1 * 1000) + " FPS: {:.3f}".format(1/s1)
            cv2.putText(frame, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255, 0, 0), 2)
            
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            
            #client.publish("person", json.dumps({"total":tcount}), retain=False)
            #client.publish('person', json.dumps({'count': current_count, 'total': count}),retain=False)
            
            ### Topic "person/duration": key of "duration" ###
            #client.publish("person/duration", json.dumps({"duration": int(s)}), retain=False)
            
            out.write(frame)
            

        ### TODO: Send the frame to the FFMPEG server ###
        #sys.stdout.buffer.write(frame)
        #sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image:
            cv2.imwrite("output.jpg", frame)
        if key_pressed == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    #sclient.disconnect()
    out.release()
    
    
def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == '__main__':
    main()