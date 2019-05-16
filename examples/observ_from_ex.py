import habitat
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

pickle_in=open("./examples/from_example_dot_py.pickle","rb")
obs_list=pickle.load(pickle_in)
ob1,ob2=obs_list[0],obs_list[1]

plt.imshow(ob1['rgb'])
plt.show()
plt.imshow(ob2['rgb'])
plt.show()



