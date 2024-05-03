import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# This code uses the flow_from_directory method of the ImageDataGenerator class 
# to create a generator that reads images from a directory and applies the rescaling 
# and validation split. The class_mode='categorical' parameter is used since the 
# images are labeled with 3 classes. The target_size parameter is used to resize 
# the images to (256, 256) and the batch_size parameter is set to 32. The fit method
#  is used to train the model on the training generator and validate it on the 
#  validation generator. The training images are splitted into training and validation 
#  sets by validation_split parameter and it will automatically split images according to the given ratio.

# define the directory that contains the training images
data_dir = 'path/to/your/image/folder'

# create an ImageDataGenerator object
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# create the training and validation generators
train_gen = datagen.flow_from_directory(data_dir, class_mode='categorical', target_size=(256,256), batch_size=32, subset='training')
val_gen = datagen.flow_from_directory(data_dir, class_mode='categorical', target_size=(256,256), batch_size=32, subset='validation')

# get the number of classes
num_classes = len(train_gen.class_indices)

# define the input shape
input_shape = (256, 256, 3)

# create the model
model = unet_model(input_shape, num_classes)

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_gen, epochs=10, validation_data=val_gen)
