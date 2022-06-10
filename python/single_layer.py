import os
import pathlib
import numpy as np
import tensorflow as tf

train_dir = pathlib.Path("/home/user/Documents/deep_learning/dataset/train/")
train_count = len(list(train_dir.glob('*/*.jpg')))

test_dir = pathlib.Path("/home/user/Documents/deep_learning/dataset/test/")
test_count = len(list(test_dir.glob('*/*.jpg')))

train_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'), shuffle=False)
train_ds = train_ds.shuffle(train_count, reshuffle_each_iteration=False)

test_ds = tf.data.Dataset.list_files(str(test_dir/'*/*'), shuffle=False)
test_ds = test_ds.shuffle(test_count, reshuffle_each_iteration=False)

img_height = 96
img_width = 96
batch_size = 32

class_names = np.array(sorted([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"]))


def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

train_ds = train_ds.map(process_path)
test_ds = test_ds.map(process_path)

num_classes = len(class_names)

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)

  return ds

train_ds = configure_for_performance(train_ds)
test_ds = configure_for_performance(test_ds)

print(train_ds)

# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(num_classes, name='last_dense')
# ])

# model.compile(
#   optimizer='adam',
#   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#   metrics=['accuracy'])

# model.fit(
#   train_ds,
#   validation_data=test_ds,
#   epochs=10
# )

# model.save('saved_model/my_model')
