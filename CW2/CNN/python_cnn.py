# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:42:47 2024

@author: yad23rju
"""
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import csv
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, optimizers, callbacks
from keras.models import Model
#from keras.layers.core import Layer
#import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
#import seaborn as sns
#from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
import math
from keras.utils import to_categorical
#from keras.utils import np_utils
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten


kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

def get_image_paths(data_path, categories, num_train_per_cat):
    num_categories = len(categories)

    # Paths for each training and test image
    train_image_paths = []
    test_image_paths = []

    # The name of the category for each training and test image
    train_labels = []
    test_labels = []

    for category in categories:
        train_category_path = os.path.join(data_path, 'train', category)
        test_category_path = os.path.join(data_path, 'test', category)

        train_images = os.listdir(train_category_path)[:num_train_per_cat]
        test_images = os.listdir(test_category_path)[:num_train_per_cat]

        # Train image paths and labels
        train_image_paths.extend([os.path.join(train_category_path, image) for image in train_images])
        train_labels.extend([category] * len(train_images))

        # Test image paths and labels
        test_image_paths.extend([os.path.join(test_category_path, image) for image in test_images])
        test_labels.extend([category] * len(test_images))

    return train_image_paths, test_image_paths, train_labels, test_labels



def my_cnn(num_epochs=30, learning_rate=0.0001, batch_size=32):
    # Define CNN architecture
    num_classes = 15
    feature_length = 64
    
    inputs = layers.Input(shape=(299, 299, 3))
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(80, (3, 3), activation='relu')(x)
    # Custom made inception layer
    branch1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    branch2 = layers.Conv2D(96, (1, 1), padding='same', activation='relu')(x)
    branch2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(branch2)
    branch3 = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x)
    branch3 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(branch3)
    x = layers.Concatenate()([branch1, branch2, branch3])
    x = layers.MaxPooling2D((8, 8), strides=1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(feature_length, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    # Define training options
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Define early stopping callback
    early_stopping = callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)
    
    # Train CNN
    #history = model.fit(x_train,labels, epochs=num_epochs, verbose=1)
    #history = model.fit(x_train,labels, epochs=num_epochs, verbose=1, callbacks=[early_stopping], batch_size=batch_size)
    #print("Done training model")

    # Save the trained CNN model
    #model.save('30_epoch_no_noise.keras')

    # Extract features using the trained CNN
    #feature_extractor = models.Model(inputs=model.input, outputs=model.layers[-2].output)
    #feature_matrix = feature_extractor.predict(train_generator)

    return model

def generate_data(train_generator):
    for x_batch, y_batch_one_hot in train_generator:
        # Assuming y_batch is a single integer representing the class
        # Convert y_batch to one-hot encoded arrays
        #y_batch_one_hot = to_categorical(y_batch, num_classes=15)
        # Yield input images and one-hot encoded labels for all three outputs
        yield x_batch, [y_batch_one_hot, y_batch_one_hot, y_batch_one_hot]

def my_cnn_2(image_paths, labels,epochs_num):
    
    num_classes = 15
    
    input_layer = Input(shape=(299, 299, 3))
    
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)
    
    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')
    
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b')
    
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    
    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')
    
    
    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(num_classes, activation='softmax', name='auxilliary_output_1')(x1)
    
    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')
    
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')
    
    x = inception_module(x,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4d')
    
    
    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(num_classes, activation='softmax', name='auxilliary_output_2')(x2)
    
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')
    
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')
    
    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')
    
    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    
    x = Dropout(0.4)(x)
    
    x = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(input_layer, [x, x1, x2], name='inception_v1')
    model.summary()
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer='adam', metrics=['accuracy'])
    # Create image data generator and normalize images
    #train_datagen = ImageDataGenerator(rescale=1./255)
    #train_generator = train_datagen.flow_from_directory(
        #data_path,
        #target_size=(299, 299),
        #batch_size=32,
        #class_mode='sparse')

    # Train CNN
    history = model.fit(image_paths, [labels, labels, labels],epochs=epochs_num,  verbose=1)
    print("Done training model")

    # Save the trained CNN model
    model.save('trained_cnn_model.h5')
    
    return model, history
    


def evaluate_cnn(model, x_test, test_labels, training_history):

    # Classify test data using the trained CNN
    predicted_labels = model.predict(x_test)
    #predicted_labels = model.predict_generator(test_generator)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    
    # Convert one-hot encoded test labels to class labels
    #true_labels = np.argmax(test_labels, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predicted_labels)
    print('Accuracy:', accuracy)

    # Plot training loss and accuracy
    plt.figure()
    plt.plot(training_history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    plt.figure()
    plt.plot(training_history.history['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.show()

    # Generate confusion matrix
    cm = confusion_matrix(test_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45)
    plt.yticks(tick_marks, categories)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return accuracy, cm,predicted_labels

def evaluate_cnn_2(model, x_test, test_labels, training_history):
    

    # Classify test data using the trained CNN
    predicted_labels = model.predict(x_test)
    predicted_labels = np.argmax(predicted_labels, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predicted_labels)
    print('Accuracy:', accuracy)

    # Plot training loss and accuracy
    plt.figure()
    plt.plot(training_history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    plt.figure()
    plt.plot(training_history.history['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.show()

    # Generate confusion matrix
    cm = confusion_matrix(test_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45)
    plt.yticks(tick_marks, categories)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return accuracy, cm

def load_and_preprocess_images(image_paths, target_size=(299, 299), horizontal_flip=False, vertical_flip=False, gauss_noise=False):
    images = []
    for path in image_paths:

        image = Image.open(path)
        
        if horizontal_flip and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Apply vertical flip if specified
        if vertical_flip and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            

        image = image.resize(target_size)
        
        if gauss_noise and np.random.rand() < 0.5:
            noise = np.random.normal(scale=40, size=(target_size[1], target_size[0], 3))

            image_array = np.array(image)
            #prevent spill over to impossible values
            image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_array)

        image = np.array(image)
        #normalise
        image = image / 255.0

        images.append(image)
    #convert list of images to numpy array
    images_array = np.array(images)
    return images_array
def encode_labels(labels):
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Fit LabelEncoder and transform labels to numerical values
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels
data_path = 'data\data'
data_path_train = 'data/data/train';
data_path_test= 'data/data/test';
#%%
categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', 
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', 
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest']

num_train_per_cat = 100
train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path, categories, num_train_per_cat)
#%%
horiz_flip = True
vert_flip = True
gauss_noise = True
print("Loading images")
#CURRENTLY TRAINING THE CNN WITH PERTURBED IMAGES THEN GIVEN THE UNPERTURBED TEST SET
x_train = load_and_preprocess_images(train_image_paths, (299,299), horiz_flip, vert_flip, gauss_noise)
x_test = load_and_preprocess_images(test_image_paths)
#Show image
plt.imshow(x_train[0])
plt.axis('off')  # Hide axes
plt.show()
#%%
encoded_train_labels = encode_labels(train_labels)
encoded_train_labels_1 = to_categorical(encoded_train_labels, 15)
encoded_test_labels = encode_labels(test_labels)
encoded_test_labels_1 = to_categorical(encoded_test_labels, 15)
#test_labels = to_categorical(test_labels, 15)
#%%
train_model = True
eval_model_on_noise = True
epochs = 30
learning_rate = 0.0001
batch_size = 32
#%%
if train_model:
    keras_model = KerasClassifier(build_fn=my_cnn, verbose=1)

    # Define the grid search parameters
    param_grid = {
        'num_epochs': [1],
        'learning_rate': [0.0001],
        'batch_size': [16,32]
    }
    
    # Create and run the grid search
    grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x_train, encoded_train_labels)
    
    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    best_model = grid_result.best_estimator_.model
else:
    print("Loading model")
    model = load_model('v_h_noise_model.keras')
    

#model, history= my_cnn_2(x_train, encoded_train_labels, epochs)
#%%
#predictions = model.predict(x_test)
#if train_model:
#    accuracy, cm,predicted_labels = evaluate_cnn(model, x_test, encoded_test_labels, history)
#%%
h_flips = [0, 1]
v_flips = [0, 1]
gauss_noises = [0, 1]
#Creat csv file and save parameters based on iterations
if eval_model_on_noise:
    with open('accuracy_results_TEST.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['Horizontal Flip', 'Vertical Flip', 'Gaussian Noise', 'Accuracy'])
    

        for horiz_flip in h_flips:
            for vert_flip in v_flips:
                for gauss_noise in gauss_noises:
                    x_test = load_and_preprocess_images(test_image_paths, (299, 299), horiz_flip, vert_flip, gauss_noise)
                    # Classify test data using the trained CNN
                    predicted_labels = model.predict(x_test)
                    predicted_labels = np.argmax(predicted_labels, axis=1)
                    
                    accuracy = accuracy_score(encoded_test_labels, predicted_labels)
                    print('Accuracy:', accuracy)
                    
                    # Write the parameter values and accuracy to the CSV file
                    writer.writerow([horiz_flip, vert_flip, gauss_noise, accuracy])
                    
                    cm = confusion_matrix(encoded_test_labels, predicted_labels)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.colorbar()
                    tick_marks = np.arange(len(categories))
                    plt.xticks(tick_marks, categories, rotation=45)
                    plt.yticks(tick_marks, categories)
                
                    fmt = 'd'
                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, format(cm[i, j], fmt),
                                     ha="center", va="center",
                                     color="white" if cm[i, j] > thresh else "black")
                
                    plt.xlabel('Predicted labels')
                    plt.ylabel('True labels')
                    plt.title('Confusion Matrix')
                    plt.tight_layout()
                    plt.show()
    

