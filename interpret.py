import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
import tensorflow as tf
import os

model = tf.keras.models.load_model("/Users/lucas/swiss_knife/zords/megazord.pb")
directory = "/Users/lucas/swiss_knife/test"

x_train = []
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".JPG") or filename.endswith(".jpg"):
        im = np.array(mpimg.imread(directory+"/"+str(filename)))
        x_train.append(im)
for i in tqdm(range(5)):

    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
             x_train[i],
             model.predict
    )

    plt.imshow(x_train[i])
    image, mask = explanation.get_image_and_mask(
             model.predict(
                  x_train[i].reshape((1,256,256,3))
             ).argmax(axis=1)[0],
             positive_only=True,
             hide_rest=False)

    plt.imshow(mark_boundaries(image, mask))
