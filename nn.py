import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import adam_v2
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import keras
print(keras.__version__)



# split test, train data
df = pd.read_csv("processed_data.csv")
attr_cols = ["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S"]
X = df[attr_cols]# attributes
y = df["Survived"] # prediction label
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

row, col = X_train.shape
print(X_train.shape)


# Convert NumPy arrays to TensorFlow tensors: tf only takes tensors
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)


# model implementation
# structure
model = keras.models.Sequential()
model.add(Dense(7, input_dim= 12, activation="relu"))
model.add(Dense(1,activation="sigmoid"))

# compile
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train,y_train, epochs= 10, batch_size= 20)


# plot confusion matrix
cm = confusion_matrix(y_test, display_labels=["Survived", "Not survived"])
cm_dis = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels= ["Survived", "Not survived"])
cm_dis.plot()
plt.show()