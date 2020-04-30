#Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from numpy import linalg as LA
import random
from scipy.signal import savgol_filter

from image_net_labels import labels_to_idx

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import tensorflow_hub as hub


#helper functions to generate the encoding
def one_hot(index, vocab_size=1000):
    index = np.asarray(index)
    if len(index.shape) == 0:
        index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    output = np.zeros((num, vocab_size), dtype=np.float32)
    output[np.arange(num), index] = 1
    return output

def interpolate_linear(v1, v2, num_steps):
    vectors = []
    for x in np.linspace(0.0, 1.0, num_steps):
        vectors.append(v2*x+v1*(1-x))
    return np.array(vectors)


def map_frames_to_latents(frame_arr):
    latent_arr = []
    for frame in frame_arr:
        flat_frame = frame.flatten().astype(np.float32)
        steps = len(flat_frame)//140 + 1
        downsampled_frame = flat_frame[::steps]
        scaled_frame = np.interp(downsampled_frame, (0, 255), (-2, 2))
        latent_arr.append(scaled_frame)

    return np.array(latent_arr)


def update_latent(z, update_vector):
    for i, val in enumerate(z):
        if val < 0:
            z[:,i] += update_vector[i]
    else:
        z[:,i] -= update_vector[i]
    z = np.clip(z, a_min=-2.0, a_max=2.0)


def scale_vals(arr):
    old_min = np.min(arr)
    old_max = np.max(arr)
    new_max = 2.0
    new_min = -2.0

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_arr = (((arr - old_min) * new_range) / old_range) + new_min

    return new_arr

#get new update directions
def new_update_dir(nv2,update_dir):
  for ni,n in enumerate(nv2):
    if n >= 2*truncation - tempo_sensitivity:
      update_dir[ni] = -1

    elif n < -2*truncation + tempo_sensitivity:
      update_dir[ni] = 1

  return update_dir

# Get new jitters
def new_jitters(jitter):
    jitters=np.zeros(140)
    for j in range(140):
        if random.uniform(0,1)<0.5:
            jitters[j]=1
        else:
            jitters[j]=1-jitter
    return jitters

def gen_y(predicts):
    label_0_txt = predicts[0][0].replace('_', ' ')
    label_1_txt = predicts[1][0].replace('_', ' ')
    label_2_txt = predicts[2][0].replace('_', ' ')
    label_0 = labels_to_idx[label_0_txt]
    label_1 = labels_to_idx[label_1_txt]
    label_2 = labels_to_idx[label_2_txt]
    label_0_weight = predicts[0][1] / 10
    label_1_weight = predicts[1][1] / 10
    label_2_weight = predicts[2][1] / 10

    y = one_hot(label_0)*label_0_weight + one_hot(label_1)*label_1_weight + one_hot(label_2)*label_2_weight
    label_txt = label_0_txt + " " + label_1_txt + " " + label_2_txt

    return y, label_txt

def top_y(predicts):
  label_0_txt = predicts[0][0].replace('_', ' ')
  label_0 = labels_to_idx[label_0_txt]
  label_0_weight = predicts[0][1] / 10

  y = one_hot(label_0) #*label_0_weight
  label_txt = label_0_txt

  return y, label_txt

#smooth class vectors
def smooth(class_vectors,smooth_factor):

    if smooth_factor==1:
        return class_vectors

    class_vectors_terp=[]
    length = int(len(class_vectors)/smooth_factor)
    for c in range(length):
        ci=c*smooth_factor
        cva=np.mean(class_vectors[int(ci):int(ci)+smooth_factor],axis=0)
        cvb=np.mean(class_vectors[int(ci)+smooth_factor:int(ci)+smooth_factor*2],axis=0)

        for j in range(smooth_factor):
            cvc = cva*(1-j/(smooth_factor-1)) + cvb*(j/(smooth_factor-1))
            class_vectors_terp.append(cvc)

    remaining = len(class_vectors) - length*smooth_factor
    cva=np.mean(class_vectors[-remaining:],axis=0)
    cvb=np.mean(class_vectors[-2*remaining:-remaining],axis=0)
    for j in range(remaining):
        cvc = cva*(1-j/(smooth_factor-1)) + cvb*(j/(smooth_factor-1))
        class_vectors_terp.append(cvc)

    return np.array(class_vectors_terp)


def create_latent_vectors_from_video(frame_arr):

  latent_arr = map_frames_to_latents(frame_arr)
  smoothed_latent = savgol_filter(latent_arr, 151, 1, mode='nearest', axis=0)

  return smoothed_latent


def create_class_vectors(predictions):
  class_vectors = []
  label_list = []

  for pred in predictions:
      label_0_txt = pred[0][0].replace('_', ' ')
      label_list.append(label_0_txt)
      label_0 = labels_to_idx[label_0_txt]

      y = one_hot(label_0)
      class_vectors.append(y)

  return smooth(class_vectors, 20), label_list


def gen_biggan_arr(latent_vectors, class_vectors, label_list):
    # Import BigGan model
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    model_size = "2)biggan-256"
    which_model = model_size.split(')')[1]
    module_path = 'https://tfhub.dev/deepmind/'+which_model+'/2'
    module = hub.Module(module_path)

    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
            for k, v in module.get_input_info_dict().items()}
    output = module(inputs)
    print ('Inputs:\n', '\n'.join(
        '{}: {}'.format(*kv) for kv in inputs.items()))
    print ('Output:', output)
    truncation = 1.0
    initializer = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(initializer)
        images = []

        for i in range(min(len(latent_vectors), len(class_vectors))):

            feed_dict = {inputs['z']: latent_vectors[i].reshape(1,140),
                         inputs['y']: class_vectors[i],
                         inputs['truncation']: truncation}

            im = sess.run(output, feed_dict=feed_dict)

            #postprocess the image
            im = np.clip(((im + 1) / 2.0) * 256, 0, 255)
            im = np.uint8(im).squeeze()
            cv2.putText(im, label_list[i], org=(5, 245), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=8)
            images.append(im)

        return images


def test_process(frame_arr, label_list):
    images = []

    for i, im in enumerate(frame_arr):
        im = cv2.resize(im, (256, 256))
        #postprocess the image
        im = np.uint8(im).squeeze()
        cv2.putText(im, label_list[i], org=(5, 245), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=8)
        images.append(im)

    return images
