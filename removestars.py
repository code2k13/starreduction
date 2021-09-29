#!/usr/bin/python

import tensorflow as tf
import model
import os
import time
from PIL import Image, ImageFilter,ImageEnhance
import random 
import numpy as np
import sys

IMG_SIZE = 1024
all_outputs = 1
pad_width = 24

args = sys.argv 
if len(args) != 3:
  print("Wrong arguments. Usage  removestars.py input_file output_file")
  print ('Argument List:', str(sys.argv)) 
  exit(1)


def process_channel(channel,pad_width,input_image_size):
    output_image =  Image.new('L', input_image_size)   
    for i in range(0,int(channel.size[0]/IMG_SIZE)):
        for j in range(0,int(channel.size[1]/IMG_SIZE)):         
            corp_rect = (i*IMG_SIZE,j*IMG_SIZE,i*IMG_SIZE+IMG_SIZE,j*IMG_SIZE+IMG_SIZE)
            current_tile = channel.crop(corp_rect)
            current_tile = current_tile.convert('L')
            blank_image =  current_tile.copy()  
            current_tile = current_tile.resize((IMG_SIZE-pad_width*2,IMG_SIZE-pad_width*2))
            blank_image.paste(current_tile,(pad_width,pad_width))
            blank_image  = np.asarray(blank_image,dtype="float16").reshape(1,IMG_SIZE,IMG_SIZE)/255
            predicted_section = G2.predict(blank_image)            
            predicted_section = predicted_section.reshape(IMG_SIZE,IMG_SIZE)*255 
            predicted_section = Image.fromarray(predicted_section).convert('L')            
            predicted_section = predicted_section.crop((pad_width,pad_width,IMG_SIZE-pad_width,IMG_SIZE-pad_width))
            predicted_section = predicted_section.resize((IMG_SIZE,IMG_SIZE))   
            output_image.paste(predicted_section, (i*IMG_SIZE,j*IMG_SIZE),  mask=None)            
    return output_image

G2 = model.Generator()
G2.load_weights("weights/weights")

source_image = Image.open(args[1])
mode = source_image.mode
size = source_image.size
max_dimension = max(size)

a,b = divmod(max_dimension,1024)
if a > 0 and b !=0 :
  max_dimension = max_dimension + b
 
input_image = source_image.crop((0,0,max_dimension,max_dimension))

if mode == 'L':  
  output = process_channel(input_image,pad_width,input_image.size)
  output = output.crop((0,0,size[0],size[1]))
  output.save(args[2])
elif mode == 'RGB' or mode == 'RGBA':
  channels = Image.Image.split(input_image)
  all_outputs  = []
  for channel in channels[0:3]:
    channel_output = process_channel(channel,pad_width,input_image.size)
    channel_output = channel_output.crop((0,0,size[0],size[1]))
    all_outputs.append(channel_output)
 
  output =Image.merge('RGB', (all_outputs[0],all_outputs[1],all_outputs[2]))
  output.save(args[2])
else:
  print("Invalid mode for input image:", mode)
  print("Only grayscale(L) and RGB(RGBA) images are supported.")
