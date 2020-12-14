# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:40:31 2020

@author: Nagesh
"""

#%%

from tkinter import *
import numpy as np
import pandas as pd
import pickle
import cv2

#%%

df = pd.read_csv("mnist_data.csv", delimiter = ",")
xy = np.array(df, dtype = 'float')

#%%
with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)


with open('mnist_pca_svm.pkl', 'rb') as f:
    model = pickle.load(f)


#%%


fontsize = 15
window1 = Tk()
window1.title("GUI")

def mnist():
    idx = np.random.randint(0, xy.shape[1])
    x = xy[idx, :-1].reshape(1, -1)
    y_pred = int(model.predict(pca.transform(x)))
    y_true = int(xy[idx, -1])
    print(idx)
    shape = (480, 480)
    x = x.reshape(28, 28)
    im = cv2.resize(x, (360, 360), interpolation = cv2.INTER_AREA)
    true_image = cv2.imread("images/" + str(y_true) + ".jpg")
    true_image = cv2.resize(true_image, shape, interpolation = cv2.INTER_AREA)
    pred_image = cv2.imread("images/" + str(y_pred) + ".jpg")
    pred_image = cv2.resize(pred_image, shape, interpolation = cv2.INTER_AREA)
    
    cv2.imshow("mnist_image", im)
    cv2.imshow("predicted_label", pred_image)
    cv2.imshow("true_label", true_image)
    
button = Button(window1, text = "ClickButton", command = mnist)
button.pack()
window1.mainloop()

