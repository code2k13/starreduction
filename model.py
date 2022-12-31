import tensorflow as tf 

IMG_SIZE = 512
OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True,strides = 2,name=''):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential(name=name)
  result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())
  return result

def upsample(filters, size, apply_dropout=False,strides = 2,name=''):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential(name=name)
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,padding='same',
                                    kernel_initializer=initializer,use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))        
  result.add(tf.keras.layers.ReLU())
  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])

  down_stack = [
    downsample(16, 5, apply_batchnorm=False,strides = 1, name='gd_1'),  # (batch_size, 128, 128, 64)
    downsample(32, 5,name='gd_2'),  # (batch_size, 64, 64, 128)
    downsample(64, 5,name='gd_3'),  # (batch_size, 32, 32, 256)
    downsample(128, 5,name='gd_4'),  # (batch_size, 16, 16, 512)
    downsample(256, 5,name='gd_5'),  # (batch_size, 8, 8, 512)
    downsample(256, 5,name='gd_6'),  # (batch_size, 4, 4, 512)
    downsample(512, 5,name='gd_7'),  # (batch_size, 2, 2, 512)
    downsample(512, 5,name='gd_8'),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 5, apply_dropout=True,name='gu_1'),  # (batch_size, 2, 2, 1024)
    upsample(256, 5, apply_dropout=True,name='gu_2'),  # (batch_size, 4, 4, 1024)
    upsample(256, 5, apply_dropout=True,name='gu_3'),  # (batch_size, 8, 8, 1024)      
    upsample(128, 5,name='gu_4'),  # (batch_size, 16, 16, 1024)
    upsample(64, 5,name='gu_5'),  # (batch_size, 32, 32, 512)
    upsample(32, 5,name='gu_6'),  # (batch_size, 64, 64, 256)
    upsample(16, 5,name='gu_7'),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,strides=1,padding='same',
                                         kernel_initializer=initializer,activation='relu')  # (batch_size, 256, 256, 3)

  x = inputs
  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)