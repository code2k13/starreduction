#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import model
from PIL import Image
import numpy as np
import sys
from tqdm import tqdm
import math


IMG_SIZE = 512
MODEL_SIZE = 1024
all_outputs = 1
pad_width = 2
total_steps = 0
progress_bar = None
current_progress= 0

args = sys.argv 
if len(args) != 3:
  print("Wrong arguments. Usage  removestars.py input_file output_file")
  print ('Argument List:', str(sys.argv)) 
  exit(1)

def process_tile(channel,i,j,pad_width,model,output_image):
  corp_rect = (i*IMG_SIZE,j*IMG_SIZE,i*IMG_SIZE+IMG_SIZE,j*IMG_SIZE+IMG_SIZE)
  current_tile = channel.crop(corp_rect)
  current_tile = current_tile.convert('L')
  blank_image =  current_tile.copy()  
  current_tile = current_tile.resize((IMG_SIZE-pad_width*2,IMG_SIZE-pad_width*2))
  blank_image.paste(current_tile,(pad_width,pad_width))
  blank_image = blank_image.resize((MODEL_SIZE,MODEL_SIZE))
  blank_image  = np.asarray(blank_image,dtype="float16").reshape(1,MODEL_SIZE,MODEL_SIZE)/512
  predicted_section = model.predict(blank_image,verbose=0)      
  predicted_section = predicted_section.reshape(MODEL_SIZE,MODEL_SIZE)*1024
  predicted_section = Image.fromarray(predicted_section).convert('L')
  predicted_section =  predicted_section.resize((IMG_SIZE,IMG_SIZE))            
  predicted_section = predicted_section.crop((pad_width,pad_width,IMG_SIZE-pad_width,IMG_SIZE-pad_width))
  predicted_section = predicted_section.resize((IMG_SIZE,IMG_SIZE))  
  output_image.paste(predicted_section, (i*IMG_SIZE,j*IMG_SIZE),  mask=None)

def process_channel(channel,pad_width,input_image_size):    
    global progress_bar,step_size,current_progress
    output_image =  Image.new('L', input_image_size)
    for i in range(0,int(channel.size[0]/IMG_SIZE)):
        for j in range(0,int(channel.size[1]/IMG_SIZE)):                   
            process_tile(channel,i,j,pad_width,G2,output_image)
            current_progress = current_progress + step_size
            if current_progress <= 100:
                progress_bar.update(round(step_size,2))
    return output_image

try:
  G2 = model.Generator()
  G2.load_weights("weights/weights")

  source_image = Image.open(args[1])
  mode = source_image.mode
  size = source_image.size
  max_dimension = max(size)

  a,b = divmod(max_dimension,IMG_SIZE)
  if b > 0 :
    max_dimension = (a+1)*IMG_SIZE
  
  input_image = source_image.crop((0,0,max_dimension,max_dimension))
  progress_bar = tqdm(total=100,leave=True)

  if mode == 'L':  
    total_steps = math.ceil((max_dimension/IMG_SIZE)**2) 
    step_size = (1/total_steps)*100
    output = process_channel(input_image,pad_width,input_image.size)
    output = output.crop((0,0,size[0],size[1]))
    output.save(args[2])
  elif mode == 'RGB' or mode == 'RGBA':
    total_steps = int(3*(max_dimension/IMG_SIZE)**2)
    step_size = (1/total_steps)*100 
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
except Exception as e:
  print(e)
