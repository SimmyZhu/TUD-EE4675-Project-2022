import os
from utils import *
from tensorflow import keras
import numpy as np
import sorted
if __name__ == "__main__":
    list_size = 1024
    data_set = []
    file_names = []
    for folder_name in os.listdir('MATLAB/Point_Cloud_Dataset_Test'):
 #       lst = os.listdir('MATLAB/Point_Cloud_Dataset_Test/' + folder_name)
  #      lst.sort()
        for i in range(1,101):
            file_name = "case" + str(i)
            file = open('MATLAB/Point_Cloud_Dataset_Test/' + folder_name + '/' + file_name + '.csv')
            data_point = np.genfromtxt('MATLAB/Point_Cloud_Dataset_Test/' + folder_name + '/' + file_name + '.csv',
                                       delimiter=',').tolist()
            # Data Normalization
   #         data_point = np.array(data_point)
 #           data_point = np.ndarray.tolist(
  #              (data_point - np.mean(data_point, axis=0)) / (np.std(data_point, axis=0) + 1e-6))
            data_point += [data_point[0]] * (list_size - len(data_point)) # keep points fixed to 1024 by seting additional values to the first point
            data_set.append(data_point)
            file_names.append(file_name)
    data_set = np.array(data_set)   

# Model Parameters
NUM_POINTS = 1024
NUM_CLASSES = 6
BATCH_SIZE = 32

# Inputs
inputs = keras.Input(shape=(NUM_POINTS, 4))

# Main Network (T-net removed)
#x = tnet(inputs, 4)
x = conv_bn(inputs, 32)
x = conv_bn(x, 32)
#x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

# Output
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

# Build the model
model = keras.models.load_model("best_model")

# Model Evaluation and Testing

pred_logits_val=model.predict(data_set)
pred_labels_val = np.argmax(pred_logits_val, axis = 1)+1
pred_labels_val=pred_labels_val.reshape(-1,1)
np.savetxt("results.csv", pred_labels_val.astype(np.int), fmt='%d')
i = 0
with open('output.csv','w') as out:
    for row in file_names:
        out.write('{0}'.format(row))
        out.write('\n')