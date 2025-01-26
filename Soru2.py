from deepface import DeepFace
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def visualize_attribute(img_path, result, attribute="age"):

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    value = result[0].get(attribute, 'not found')
    if attribute == "emotion" or attribute == "gender":
        text = f"{attribute.capitalize()}\n"
        for k, v in value.items():
            text += f"{k.capitalize()}: {round(float(v), 2)}\n"
    else:
        if isinstance(value, (float, np.float32, np.float64)):
            value = round(float(value), 2)
        text = f"{attribute.capitalize()}: {value}"

    print(text)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(text)
    plt.show()

img_path = r"face4.jpg";
actions = ['age', 'gender', 'emotion']
result = DeepFace.analyze(img_path=img_path, actions=actions)

for a in actions:
    visualize_attribute(img_path, result, a)

# Age: 32

# Gender
# Woman: 100.0
# Man: 0.0

# Emotion
# Angry: 0.0
# Disgust: 0.0
# Fear: 0.0
# Happy: 99.41
# Sad: 0.01
# Surprise: 0.0
# Neutral: 0.57

