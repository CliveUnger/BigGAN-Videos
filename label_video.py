#Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils, mobilenet
#import tensorflow as tf
#print("Using TensorFlow Version: ", tf.version.VERSION)

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
#tf.disable_v2_behavior()


# Read video that is uploaded
def read_video(file_name):
    #nparr = np.fromstring(file_name.read(), numpy.uint8) #request.files['file'].read()
    #img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    cap = cv2.VideoCapture(file_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_arr = []

    #Convert video resolution to 244x244
    index = 0
    cap.set(3, 244)
    cap.set(4, 244)
    while True:
        ret, frame = cap.read()
        if ret == True:
            img = cv2.resize(frame, (224, 224))
            video_arr.append(img)
        else:
            break

    print("Video Length: ", len(video_arr))
    print("Video Shape: ", video_arr[0].shape)
    return video_arr, fps

def make_frame_predictions(video_arr):
    #Use mobilenet to generate top three predictions for each frame
    #Will take a few minutes to run
    mobilenet_model = mobilenet.MobileNet()
    video_predictions_1 = []
    video_predictions_2 = []
    video_predictions_3 = []
    for i in range(0, len(video_arr)):
        img_array = np.expand_dims(video_arr[i], axis=0)
        pImg = mobilenet.preprocess_input(img_array)
        prediction = mobilenet_model.predict(pImg)
        results = imagenet_utils.decode_predictions(prediction)
        if i == 0:
          print(results)
        video_predictions_1.append(results[0][0][1])#, results[0][0][2]))
        video_predictions_2.append(results[0][1][1])#, results[0][1][2]))
        video_predictions_3.append(results[0][2][1])#, results[0][2][2]))

    #Combine into single array of tuples
    video_predictions = [None] * len(video_predictions_1)
    for i in range(0, len(video_predictions_1)):
        video_predictions[i] = [(video_predictions_1[i]),(video_predictions_2[i]),(video_predictions_3[i])]

    return video_predictions

def average_predictions(video_predictions):
#Average over 10 frames, order top 3 predictions
    averaged_video_predictions = [None] * len(video_predictions)

    for i in range(0, len(video_predictions)):
        top_match = {}
        min_index = i - 5
        max_index = i + 5
        if (i<5):
            min_index = 0
        if (i > (len(video_predictions) - 5)):
            max_index = len(video_predictions) - 1

        for j in range(min_index, max_index):
            if (top_match.get(video_predictions[j][0])):
                top_match[video_predictions[j][0]] = top_match[video_predictions[j][0]] + 1
            else:
                top_match[video_predictions[j][0]] = 1

            if (top_match.get(video_predictions[j][1])):
                top_match[video_predictions[j][1]] = top_match[video_predictions[j][1]] + 1
            else:
                top_match[video_predictions[j][1]] = 1

            if (top_match.get(video_predictions[j][2])):
                top_match[video_predictions[j][2]] = top_match[video_predictions[j][2]] + 1
            else:
                top_match[video_predictions[j][2]] = 1

        #Order the top matches by frequency, then save it into the averaged array
        sorted_top_match = sorted(top_match.items(), key=lambda kv: kv[1], reverse = True)
        try:
            if (sorted_top_match[0] and sorted_top_match[1] and sorted_top_match[2]):
                averaged_video_predictions[i] = [(sorted_top_match[0]), (sorted_top_match[1]), (sorted_top_match[2])]
        except:
            try:
                if (sorted_top_match[0] and sorted_top_match[1]):
                    averaged_video_predictions[i] = [(sorted_top_match[0]), (sorted_top_match[1])]
            except:
                if (sorted_top_match[0]):
                    averaged_video_predictions[i] = [(sorted_top_match[0])]

    return averaged_video_predictions
