#The libraries
import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

#Setup the data
data = [] #All images go here
label = [] #The label for the images. 0 for control, 1 for dyslexia
folders = ['Control', 'Dyslexia'] #The folders that exists in the image folder
label_val = 0.0
for file in folders:
    path = os.path.join('D:\MMU\FYP\Dataset\CroppedTF3', file)
    for img in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        data.append(image_array)
        label.append(label_val)
    label_val = 1.0

#Convert to numpy array
data = np.asarray(data)
label = np.asarray(label)

#Splitting the images along with the labels into training validation and test sets
'''
test_size = 0.2 means 80% will go to the train set while 20% go to the test set(well in this case, a giant remain set for further splitting)
test_size = 0.5 means 50% will go to the validation set while 50% go to the test set
random_state controls the shuffling applied to the data before applying the split. Pass an integer for reproducible output across multiple function calls.
'''
x_train, x_remain, y_train, y_remain = sklearn.model_selection.train_test_split(data, label, test_size = 0.2, random_state = 123)
x_val, x_test, y_val, y_test = sklearn.model_selection.train_test_split(x_remain, y_remain, test_size = 0.5, random_state = 123)

'''
#To check the shape of the sets
print('x_train shape: ', x_train.shape)
print('Train samples: ', x_train.shape[0])
print('Validate samples: ', x_val.shape[0])
print('Test samples: ', x_test.shape[0])
print('y_train shape: ', y_train.shape)
print('y_val samples: ', y_val.shape)
print('y_test samples: ', y_test.shape)
'''

#Setup the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size = (5, 5), activation = 'relu', input_shape = (479, 408, 3)), #Start with a bigger kernel due to size of the image
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(0.25), #0.25 means 25%
    tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'), #ReLU is used because allows backpropagation of the error and learning to continue, even for high values of the input to the activation function
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'), 
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1, activation = 'sigmoid') #sigmoid is used because this is a binary classification (dyslexic or not)
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting & testing the model
history = model.fit(x_train, y_train, batch_size = 50, epochs = 30, verbose = 1, validation_data = (x_val, y_val))
#model.save('D:\MMU\FYP\Code\Python\Marsha_Dyslexia')

#Confusion Matrix
model = tf.keras.models.load_model('D:\MMU\FYP\Code\Python\Marsha_Dyslexia')
predictions = model.predict(x_train)
cm = tf.math.confusion_matrix(predictions, y_train)
#print(cm)
plt.figure(figsize = (10, 8))
sns.heatmap(cm, xticklabels = folders, yticklabels = folders, annot = True, fmt = '')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

#Plot the accuracy and loss of the training process of the model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''
#to load a model
#model = tf.keras.models.load_model('D:\MMU\FYP\Code\Python\Marsha_Dyslexia')

score = model.evaluate(x_test, y_test, verbose = 0) #Evaluating the model
print('Test score', score[0])
print('Test accuracy', score[1])

#Confusion matrix
predictions = model.predict(x_test)
print(tf.math.confusion_matrix(predictions, y_test))
'''
