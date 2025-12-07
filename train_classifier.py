"""Trains the gesture recognition model"""

# importing modules
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

'''
Pickle module can save any python object like classes, functions, lists, dicts etc into a file and load it whenever needed in the exact state it was saved it.
Taking a Python object -> freezing it -> storing -> unfreezing later.
'''

Landmark_info,  gestures= [], []

with open("dataset.csv") as f:
    reader = csv.reader(f) # creating a reader object on the opened csv file.
    next(reader)  # skip header
    for row in reader:
        gestures.append(row[0]) # appending the gesture name in y, as row[0] contains the gesture name
        Landmark_info.append(list(map(float, row[1:]))) # appending the numerical coordinates and features of the 21 points (63 values) to X

Landmark_info, gestures = np.array(Landmark_info), np.array(gestures) # converting the lists to arrays as ML models work on arrays not on lists because of their benefits

le = LabelEncoder() # converts classes into numbers (basically converts text into number as ML models cannot understand text but need numbers to work with).
gestures_enc = le.fit_transform(gestures) # converts the gesture labels in y into numerical values.

Landmark_info_train, Landmark_info_test, gestures_train, gestures_test = train_test_split(Landmark_info, gestures_enc, test_size=0.2) # splits the available dataset into two parts -> training (80%) and testing (20%)
'''
Logic->
The dataset we loaded was first loaded in two different parts, Landmark_info and gestures, where Landmark_info was the numerical representation of the coordinated of the various points on the palm 
which were obtained upon hand detection, and the gestures list contains the name of the corresponding gesture the given numerical scenerio of points correspond to.

So now we are splitting this dataset into 2 main parts (2 parts of Landmark_info (Landmark_info_train, Landmark_info_test) and 2 parts of gestures (gestures_train, gestures_test))

NOTE:- BASICALLY: “When the input looks like this, the correct output should be that.”
'''


neural_network = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300) # creates a neural network with two layers (one with 128 neurons and the next one with 64 neurons) and max iterations of 300
'''
Logic-> Values used come from trial and error or from documentations.
MLP - Multi Level Perception -> a neural network generally used for classification
This basically creates a neural network with the given classification.

Important things- More layers != better accuracy
max iterations -> maximum number of times the model will rerun itself to train and test and adjust its weights.
-> too little: too little time for the model to train
-> too much: too much time, thus, model spends time not learning anything useful.
'''
neural_network.fit(Landmark_info_train, gestures_train) # training the model -> fitting is similar to training in normal words

print("Accuracy:", neural_network.score(Landmark_info_test, gestures_test)*100, "%") # printing the result of the model's test.

pickle.dump({"model": neural_network, "encoder": le}, open("model.pkl", "wb")) # storing the trained neural network in a pkl file.
print("Saved model.pkl")

