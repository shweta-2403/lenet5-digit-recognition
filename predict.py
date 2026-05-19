import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("lenet5_model.h5")

# Load image
image = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)

# Resize image
image = cv2.resize(image, (28, 28))

# Normalize image
image = image / 255.0

# Reshape for CNN
image = image.reshape(1, 28, 28, 1)

# Predict digit
prediction = model.predict(image)

digit = np.argmax(prediction)

print("Predicted Digit:", digit)