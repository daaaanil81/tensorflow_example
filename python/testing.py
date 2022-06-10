
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

img_height = 96
img_width = 96
batch_size = 32

train_dir = pathlib.Path("/home/user/Documents/deep_learining/dataset/train/")

class_names = np.array(sorted([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"]))

model = tf.keras.models.load_model('saved_model/my_model')

sunflower_path = "/home/user/Downloads/test.jpg"

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(class_names)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


model.summary()

