from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical


model = Sequential()
model.add(Dense(64, input_dim = 4, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model.summary())

model.fit(X, to_categorical(y), epochs = 100, verbose = 1)