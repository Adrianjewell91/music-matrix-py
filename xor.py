import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense
import tensorflow

# XOR inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Create a sequential model
model = Sequential()

# Add layers
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(inputs, outputs, epochs=5000, verbose=0)

# Evaluate the model
print(model.evaluate(inputs, outputs))

# WOW one time it actually worked, the other times idk.
#1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step - accuracy: 1.0000 - loss: 0.0227
# [0.022708799690008163, 1.0]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
# [[0.03467137]
#  [0.9883022 ]
#  [0.9915409 ]
#  [0.03467137]]

print(model.predict(inputs))

# https://github.com/Adrianjewell91/theorizer/blob/master/neural_net/views.py
# https://github.com/Polaris000/BlogCode/blob/main/XOR_Perceptron/xorperceptron.ipynb

