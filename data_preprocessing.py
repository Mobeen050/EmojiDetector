import tensorflow as tf

IMG = 96
BATCH = 64

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True,              # safe for many emojis
    fill_mode="nearest"
).flow_from_directory(
    "emoji_dataset/train",
    target_size=(IMG, IMG),
    batch_size=BATCH,
    class_mode="categorical"
)

val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.
).flow_from_directory(
    "emoji_dataset/val",
    target_size=(IMG, IMG),
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)
