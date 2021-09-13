#!/usr/bin/python

import tensorflow as tf
import os
import time
from PIL import Image, ImageFilter,ImageEnhance
import random 
import numpy as np
import sys

IMG_SIZE = 1024
OUTPUT_CHANNELS = 1



args = sys.argv 
if len(args) != 3:
  print("Wrong arguments. Usage  removestars.py input_file output_file")
  print ('Argument List:', str(sys.argv)) 
  exit(1)

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
        
  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
    
  inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    #print(up_stack,skips)
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


G2 = Generator()
G2.load_weights("weights/weights")
im_web = Image.open(args[1])


mode = im_web.mode
size = im_web.size
max_dimension = max(size)
input_image = im_web.crop((0,0,max_dimension,max_dimension))
input_image = input_image.resize((IMG_SIZE,IMG_SIZE))
if mode == 'L':  
  sample = np.asarray(input_image,dtype="float32").reshape(1,IMG_SIZE,IMG_SIZE,1)/255
  sample_prediction = G2(sample, training=False)
  img  = sample_prediction.numpy()
  img.resize(IMG_SIZE,IMG_SIZE)
  im = Image.fromarray(img*255)
  im  = im.convert("L")
  im = im.crop((0,0,size[0]*1024/max_dimension,size[1]*1024/max_dimension))
  if max_dimension < 1024:
    im = im.resize((size[0],size[1]))
  im.save(args[2])
elif mode == 'RGB':
  channels = Image.Image.split(input_image)
  output_channels  = []
  for channel in channels:
    sample = np.asarray(channel,dtype="float32").reshape(1,IMG_SIZE,IMG_SIZE,1)/255
    sample_prediction = G2(sample, training=False)
    img  = sample_prediction.numpy()
    img.resize(IMG_SIZE,IMG_SIZE)
    im = Image.fromarray(img*255)
    im  = im.convert("L")
    im = im.crop((0,0,size[0]*1024/max_dimension,size[1]*1024/max_dimension))
    if max_dimension < 1024:
      im = im.resize((size[0],size[1]))
    output_channels.append(im)
  output =Image.merge('RGB', (output_channels[0],output_channels[1],output_channels[2]))
  output.save(args[2])
else:
  print("Invalid mode for input image:", mode)
  print("Only grayscale(L) and RGB(RGB) images are supported. Images with alpha channel are not supported.")
