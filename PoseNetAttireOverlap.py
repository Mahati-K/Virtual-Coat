# https://medium.com/roonyx/pose-estimation-and-matching-with-tensorflow-lite-posenet-model-ea2e9249abbd
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
# from google.colab.patches import cv2_imshow
import math

def getPosePoints(img):

    # Steps for running:
    # 1. Download PoseNet model from https://www.tensorflow.org/lite/models/pose_estimation/overview
    # 2. Choose your template and target image to process
    # 3. Specify paths


    model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
    # template_path = imgPath

    # Load TFLite model and allocate tensors (memory usage method reducing latency)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors information from the model file
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    template_image_src = img
    src_tepml_width, src_templ_height, _ = template_image_src.shape 
    template_image = cv.resize(template_image_src, (width, height))
    # cv.imshow('Original', template_image)

    # can be used later to draw keypoints on the source image (before resizing)
    templ_ratio_width = src_tepml_width/width
    templ_ratio_height = src_templ_height/height

    # add a new dimension to match model's input
    template_input = np.expand_dims(template_image.copy(), axis=0)

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # Floating point models offer the best accuracy, at the expense of model size 
    # and performance. GPU acceleration requires the use of floating point models.

    # Brings input values to range from 0 to 1

    if floating_model:
        template_input = (np.float32(template_input) - 127.5) / 127.5

    # Process template image
    # Sets the value of the input tensor
    interpreter.set_tensor(input_details[0]['index'], template_input)
    # Runs the computation
    interpreter.invoke()
    # Extract output data from the interpreter
    template_output_data = interpreter.get_tensor(output_details[0]['index'])
    template_offset_data = interpreter.get_tensor(output_details[1]['index'])
    # Getting rid of the extra dimension
    template_heatmaps = np.squeeze(template_output_data)
    template_offsets = np.squeeze(template_offset_data)
    # print("template_heatmaps' shape:", template_heatmaps.shape)
    # print("template_offsets' shape:", template_offsets.shape)

    # The output consist of 2 parts:
    # - heatmaps (9,9,17) - corresponds to the probability of appearance of 
    # each keypoint in the particular part of the image (9,9)(without applying sigmoid 
    # function). Is used to locate the approximate position of the joint
    # - offset vectors (9,9,34) is called offset vectors. Is used for more exact
    #  calculation of the keypoint's position. First 17 of the third dimension correspond
    # to the x coordinates and the second 17 of them correspond to the y coordinates

    def parse_output(heatmap_data,offset_data, threshold):

    # Input:
    #     heatmap_data - hetmaps for an image. Three dimension array
    #     offset_data - offset vectors for an image. Three dimension array
    #     threshold - probability threshold for the keypoints. Scalar value
    # Output:
    #     array with coordinates of the keypoints and flags for those that have
    #     low probability

        joint_num = heatmap_data.shape[-1]
        pose_kps = np.zeros((joint_num,3), np.uint32)

        for i in range(heatmap_data.shape[-1]):

            joint_heatmap = heatmap_data[...,i]
            max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
            remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
            pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
            pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
            max_prob = np.max(joint_heatmap)

            if max_prob > threshold:
                if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
                    pose_kps[i,2] = 1

        return pose_kps

    def draw_kps(show_img,kps, ratio=None):
        for i in range(5,kps.shape[0]):
            if kps[i,2]:
                if isinstance(ratio, tuple):
                    cv.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(255,0,0),round(int(1*ratio[1])))
                    cv.putText(show_img, "{}".format(i), (int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv.LINE_AA)
                    continue
                cv.circle(show_img,(kps[i,1],kps[i,0]),2,(255,0,0),-1)
                cv.putText(show_img, "{}".format(i), (kps[i,1],kps[i,0]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv.LINE_AA)
        return show_img

    template_show = np.squeeze((template_input.copy()*127.5+127.5)/255.0)
    template_show = np.array(template_show*255,np.uint8)
    template_kps = parse_output(template_heatmaps,template_offsets,0.3)
    ratio=(templ_ratio_width, templ_ratio_height)


    shoulderWidth = int((template_kps[5,1] - template_kps[6,1])*ratio[1])
    chestHeight = int((template_kps[11,0] - template_kps[5,0])*ratio[0])
    leftShoulder = (int(template_kps[6,1]*ratio[1]),int(template_kps[6,0]*ratio[0]))
    rightShoulder = (int(template_kps[5,1]*ratio[1]),int(template_kps[5,0]*ratio[0]))
    shoulderPoints={"shoulderWidth": shoulderWidth, 
                    "leftShoulder": leftShoulder, 
                    "rightShoulder": rightShoulder,
                    "chestHeight": chestHeight}
    # print(shoulderPoints)

    # cv.imshow('Points', draw_kps(template_show.copy(),template_kps))
    # cv.imshow('Points', draw_kps(template_image_src,template_kps, ratio))
    # cv.waitKey()

    return shoulderPoints

def getAttireOverlap(body, coat):
    # bImg=cv.imread(body)
    cImg=cv.imread(coat)
    bImg = body

    person = getPosePoints(bImg)
    suit = getPosePoints(cImg)

    scaleX = person['shoulderWidth']/suit['shoulderWidth']*1.1
    scaleY = person['chestHeight']/suit['chestHeight']#*1.15
 
    x_offset=person['leftShoulder'][0]-int(suit['leftShoulder'][0]*scaleX)-10
    y_offset=person['leftShoulder'][1]-int(suit['leftShoulder'][1]*scaleY)-40

    suit = cv.imread(coat, -1)
    try:
        suit = cv.resize(suit, 
                            (int(suit.shape[1] * scaleX), int(suit.shape[0] * scaleX)), 
                            interpolation = cv.INTER_AREA)
    except:
        print("Move farther from camera")
        img=bImg
    else:
        y1, y2 = y_offset, y_offset + suit.shape[0]
        x1, x2 = x_offset, x_offset + suit.shape[1]

        if(not(x1>0 and x2>0 and y1>0 and y2>0)):
            img = bImg
                    
        else:
 
            if(y2>bImg.shape[0]):
                y2=min(y2, bImg.shape[0])
                suit = suit[0:y2-y1, 0:suit.shape[1], :]

            if(x1<0):
                print("x1<0: ", x1)
                suit = suit[0:suit.shape[0], abs(x1):suit.shape[1], :]

            if(x2>bImg.shape[1]):
                print("x2>limit", x2)
                x2=min(x2, bImg.shape[1])
                suit = suit[0:suit.shape[0], 0:x2-x1, :]
 

            # print("x1, x2, y1, y2, bImg.shape[0], suit.shape[0]: ",x1, x2, y1, y2, bImg.shape[0], suit.shape[0])

            alpha_s = suit[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            img = body

            try:
                for c in range(0, 3):
                    img[y1:y2, x1:x2, c] = (alpha_s * suit[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
            except:
                img=body
                print("Come in front of the camera")

            img = cv.resize(img, (int(bImg.shape[1]), int(bImg.shape[0])), interpolation = cv.INTER_AREA)

        cv.imshow('Attire Overlap',img)

# https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481
def captureVideo(suit):
    import cv2
    vidcap = cv.VideoCapture(0,cv.CAP_DSHOW)
    def getFrame(sec):
        vidcap.set(cv.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            # getPosePoints(image)
            getAttireOverlap(image, suit)
        return hasFrames
    sec = 0
    frameRate = 0.5 #//it will capture image in each 0.5 second
    while getFrame(sec):
        sec = round((sec + frameRate), 2)
        if cv.waitKey(10) == 27:
            break
        

captureVideo("CoatSnippet.png")    
