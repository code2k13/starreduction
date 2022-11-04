import tensorflow as tf
import model
import tensorflowjs as tfjs

G2 = model.Generator()
G2.trainable = False
G2.load_weights("weights/weights")

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(G2) # path to the SavedModel directory
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

tfjs.converters.save_keras_model(G2, "model_js")