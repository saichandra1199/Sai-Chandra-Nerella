import tensorflow as tf
from tensorflow import keras
import os
from tensorflow import math
import keras.backend as K
from tensorflow import math
import sys

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"] = "0";

img_width = 224
img_height = 224
train_data_dir = 'path to train folder '
valid_data_dir = 'path to validation ot test folder'
model_path = "path to base model "
epochs=5
steps_per_epoch =6
validation_steps =20
loss='binary_crossentropy'
#def los(actual_l,pridicted_l):
#	loss=keras.losses.binary_crossentropy(actual_l, pridicted_l, from_logits=True)
#	d=math.subtract(loss,0.3)
#	loss=tf.keras.backend.maximum(0.0,d)
#	print(loss)
#	print_op = tf.print(loss,actual_l,pridicted_l, output_stream = 'file://loss.txt',summarize=-1)
#	with tf.control_dependencies([print_op]):
#		return K.identity(loss)
	#return loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_directory(directory=train_data_dir,
											   target_size=(img_width,img_height),
											   classes=['Occlusion','Proper'],
											   class_mode='binary',
											   batch_size=16,interpolation='lanczos')

validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
											   target_size=(img_width,img_height),
											   classes=['Occlusion','Proper'],
											   class_mode='binary',
											   batch_size=1,interpolation='lanczos')




model =tf.keras.models.load_model(model_path)



model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])

print('model complied!!')

print('started training....')
training = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch,epochs=epochs,validation_data=validation_generator,validation_steps=validation_steps)

print('training finished!!')

print('saving weights to h5')

model.save('modelname.h5',include_optimizer=False)

print('all weights saved successfully !!')


