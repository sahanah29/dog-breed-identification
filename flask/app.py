from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np


breeds = ['German Shepherd', 'Labrador Retriever',
          'Malamute', 'Old English Sheepdog', 'Shih Tzu']
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  imagefile=request.files['imagefile']
  image_path='./images/' + imagefile.filename
  imagefile.save(image_path)

  ds = tf.keras.utils.image_dataset_from_directory(
    './images',
    label_mode=None,
    color_mode='grayscale',
    batch_size=1,
    image_size=(224,224),
  )
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),

    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax') # maybe try sigmoid/softmax
  ])
  # model = tf.keras.models.load_model('dog_classification_model.h5')
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )
  model.load_weights('model_weights.h5')
  pred = model.predict(ds)
  breedIndex = np.argmax(pred, axis = - 1)[0]
  print(breedIndex, pred)
  label = breeds[breedIndex]
  chance = pred[0][breedIndex]
  classification = 'Predicted: %s with chance %.2f' % (label, chance)

  return render_template('index.html', prediction=classification)

if __name__ == '__main__':
  app.run(port=3000, debug=True)