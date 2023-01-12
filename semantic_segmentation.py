

#%%
#Import packages
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import callbacks
import tensorflow as tf
import numpy as np
import cv2, os
import datetime

#%%
#1.0. Data perparation
TEST_PATH = os.path.join('data-science-bowl-2018-2','test')
TRAIN_PATH = os.path.join('data-science-bowl-2018-2','train')
test_image_dir = os.path.join(TEST_PATH,'inputs')
test_mask_dir = os.path.join(TEST_PATH,'masks')
train_image_dir = os.path.join(TRAIN_PATH,'inputs')
train_mask_dir = os.path.join(TRAIN_PATH,'masks')

#%%
#2.0. Prepare empty list to hold the data
images = []
masks = []
#%%
#2.1. Load the image
def load_images(file_path):
    for image_file in os.listdir(file_path):
        img = cv2.imread(os.path.join(file_path,image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(128,128))
        images.append(img)
    return images

train_images = load_images(train_image_dir)
test_images = load_images(test_image_dir)

#%%
#2.2. Load the mask list

def load_masks(file_path):
    for mask_file in os.listdir(file_path):
        mask = cv2.imread(os.path.join(file_path,mask_file),cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(128,128))
        masks.append(mask)
    return masks

train_masks = load_masks(train_mask_dir)
test_masks = load_masks(test_mask_dir)
#%%
#2.3. Convert the list into numpy array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

# %%
#2.4. Visualize some pictures as example
#for images
plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    img_plot = train_images[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()   

#for masks
plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    mask_plot = train_masks[i]
    plt.imshow(mask_plot, cmap='gray')
    plt.axis('off')
plt.show()  
# %%
#3.0. Data preprocessing
#3.1. Expand the mask dimension
train_masks_np_exp = np.expand_dims(train_masks_np,axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np,axis=-1)
#Check the mask output
print(np.unique(train_masks[0]))

#%%
#3.2. Change the mask value (Encode into numerical encoding, binary 1 and 0)
converted_masks_train = np.ceil(train_masks_np_exp/255)
converted_masks_test = np.ceil(test_masks_np_exp/255)
converted_masks_train = 1 - converted_masks_train
converted_masks_test = 1 - converted_masks_test

#%%
#3.3. Normalize the images
converted_images_train = train_images_np / 255.0
converted_images_test = test_images_np/255.0

#%%
#4.0. Do train-validation split
from sklearn.model_selection import train_test_split

SEED = 12345
x_train,x_val,y_train,y_val = train_test_split(converted_images_train,converted_masks_train,test_size=0.2,random_state=SEED)

#%%
#5.0. Convert the numpy array into tensor slice
train_x = tf.data.Dataset.from_tensor_slices(x_train)
val_x = tf.data.Dataset.from_tensor_slices(x_val)
train_y = tf.data.Dataset.from_tensor_slices(y_train)
val_y = tf.data.Dataset.from_tensor_slices(y_val)
test_x = tf.data.Dataset.from_tensor_slices(converted_images_test)
test_y = tf.data.Dataset.from_tensor_slices(converted_masks_test)

#%%
#6.0. Zip the tensor slice into ZipDataset
train = tf.data.Dataset.zip((train_x,train_y))
val = tf.data.Dataset.zip((val_x,val_y))
test = tf.data.Dataset.zip((test_x,test_y))

#%%
#7. Define data augmentation pipeline as a single layer through subclassing
class Augment(keras.layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = keras.layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = keras.layers.RandomFlip(mode='horizontal',seed=seed)

    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels
#%%
#8.0. Convert into PrefetchDataset
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 800//BATCH_SIZE
VALIDATION_STEPS = 200//BATCH_SIZE

train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size=AUTOTUNE)

val = val.batch(BATCH_SIZE).repeat()
val = val.prefetch(buffer_size=AUTOTUNE)

test = test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# %%
#9. Visualize some pictures as example
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]),cmap='gray')
    plt.show()

for images,masks in train.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])
#%%


#10. Create the model
#10.1 Use a pretrained as feature extractor
base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#10.2. Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#10.3. Instantiate the feature extractor
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

#10.4. Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

#10.5. Use functional API to construct the entire U-net
def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128,128,3])
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Build the upsampling path and establish the concatenation
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x,skip])
    
    #Use a transpose convolution layer to perform the last upsampling, this will become the output layer
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)
    
#10.6. Use the function to create the model
OUTPUT_CLASSES = 2
model = unet_model(output_channels=OUTPUT_CLASSES)
#Model Architechture
tf.keras.utils.plot_model(model, show_shapes=True)

#11. Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
model.summary()


# %%
#12. Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions()

#%%
#13. Create a callback function to make use of the show_predictions function
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))
#%%
#13.1. TensorBoardLog
log_path = os.path.join('log_dir','asses4',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = tf.keras.callbacks.TensorBoard(log_path,histogram_freq=1,profile_batch=0)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=0)

#%%
#14. Model training
EPOCH = 50

history = model.fit(train,epochs=EPOCH,steps_per_epoch=STEPS_PER_EPOCH,batch_size=BATCH_SIZE,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=val,
                    callbacks=[DisplayCallback(),tb,es])

#%%
#Test evaluation
test_loss, test_accuracy = model.evaluate(test)
print(f"Test loss = {test_loss}")
print(f"Test accuracy = {test_accuracy}")

#%%
#Deploy model
show_predictions(test,3)

# %%
#Model Saving
# to save trained model
model.save("model.h5")
#%%

# %%
#15. Continue the model training because previous train seem undertrain
fine_tune_epoch = 20
total_epoch = EPOCH + fine_tune_epoch

# Follow up from the previous model training
history = model.fit(train,epochs=total_epoch,initial_epoch=history.epoch[-1],steps_per_epoch=STEPS_PER_EPOCH,batch_size=BATCH_SIZE,validation_steps=VALIDATION_STEPS,validation_data=val,callbacks=[DisplayCallback(),tb,es])
# %%
#16. Evaluate the final model
test_loss, test_accuracy = model.evaluate(test)
print(f"Test loss = {test_loss}")
print(f"Test accuracy = {test_accuracy}")
# %%
#Model Saving
# to save trained model
model.save("model.h5")
#%%