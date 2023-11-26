import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

# Set your dataset paths accordingly
train_path = r"C:\Users\LENOVO\OneDrive\Desktop\Weather forecasting using CNN\DATASET\training"
valid_path = r"C:\Users\LENOVO\OneDrive\Desktop\Weather forecasting using CNN\DATASET\validation"
test_path = r"C:\Users\LENOVO\OneDrive\Desktop\Weather forecasting using CNN\DATASET\test"

BATCH_SIZE = 32
IMAGE_SIZE = (150, 150)
num_classes = 11  # Adjust based on your weather categories

# ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Create a CNN model
cnn_model = Sequential([
    VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
history = cnn_model.fit(train_generator, validation_data=valid_generator, epochs=20)

# Plot training curves
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Evaluate the model on a test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_scores = cnn_model.evaluate(test_generator)
print('Test Loss:', test_scores[0])
print('Test Accuracy:', test_scores[1])

# Data Augmentation for Forecasting
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Example usage of data augmentation for forecasting
img = load_img("path/to/sample/image.jpg")  # Replace with the path to your sample image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='forecast', save_format='jpg'):
    i += 1
    if i > 20:
        break

# Continue with the rest of your code...

# Model 2 (ResNet) Architecture
keras.backend.clear_session()
input_layer = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
conv_layer = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
conv_layer = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv_layer)
pooling_layer = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_layer)
conv_layer = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(pooling_layer)
conv_layer = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv_layer)
pooling_layer = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_layer)
flatten = keras.layers.Flatten()(pooling_layer)
dense = keras.layers.Dense(200, activation="relu")(flatten)
dropout = keras.layers.Dropout(0.5)(dense)
dense = keras.layers.Dense(100, activation="relu")(dropout)
dropout = keras.layers.Dropout(0.2)(dense)
classifier = keras.layers.Dense(num_classes, activation="softmax")(dropout)
model2 = keras.Model(inputs=input_layer, outputs=classifier)
opt = keras.optimizers.Adam(learning_rate=0.001)
model2.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

# Model 3 (VGG16-based Transfer Learning) Architecture
vgg16 = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
keras.backend.clear_session()
for layer in vgg16.layers:
    layer.trainable = False

model_t = tf.keras.models.Sequential([
    vgg16,
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")
])
model_t.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model 4 (ResNet-based Transfer Learning) Architecture
resnet = keras.applications.ResNet152V2(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
keras.backend.clear_session()
for layer in resnet.layers:
    layer.trainable = False

model_res = tf.keras.models.Sequential([
    resnet,
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")
])
model_res.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training Models
history2 = model2.fit(train_generator, validation_data=valid_generator, epochs=20)
history3 = model_t.fit(train_generator, validation_data=valid_generator, epochs=20)
history4 = model_res.fit(train_generator, validation_data=valid_generator, epochs=20)

# Plotting Results
def plot_curves(history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')

# Plotting Model 2 Results
plot_curves(history2)

# Plotting Model 3 Results
plot_curves(history3)

# Plotting Model 4 Results
plot_curves(history4)

