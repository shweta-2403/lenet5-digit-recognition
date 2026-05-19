import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build LeNet-5 model
model = Sequential()

model.add(Conv2D(6, kernel_size=(5,5),
                 activation='relu',
                 input_shape=(28,28,1)))

model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(16, kernel_size=(5,5),
                 activation='relu'))

model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))

model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    x_train,
    y_train,
    epochs=2,
    batch_size=128,
    validation_data=(x_test, y_test)
)

# Save model
model.save("lenet5_model.h5")

print("Model trained and saved successfully!")
