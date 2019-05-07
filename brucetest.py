
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

pickle_in=open("from_evaluate.pickle","rb")

foo=pickle.load(pickle_in)

sensor=foo[0]
rollouts=foo[1]



rollouts.observations[sensor][2]

images=rollouts.observations[sensor].cpu().numpy()[:,0,:,:,:]/255

for i in range(11):
    plt.imshow(images[2])

