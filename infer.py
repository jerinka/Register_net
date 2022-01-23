import tensorflow as tf
import keras
import cv2
import numpy as np
import config as cfg
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def show_fused(im1,im2,win='fused',time=0):
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    fused = np.zeros_like(im1)
    fused[:,:,1] = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    fused[:,:,2] = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    cv2.imshow(win, fused)
    cv2.waitKey(time)

reg_cnn = tf.keras.models.load_model(cfg.checkpoint_path)

inputdim = reg_cnn.inputs[0].shape[1:4].as_list()

im1 = cv2.imread('samples/2_.png',1)
im2 = cv2.imread('samples/2_x_7__y_-10.png',1)

xin=[[],[]]
im11=np.expand_dims(im1,0)
im22=np.expand_dims(im2,0)
xin[0].append(im11/255.0)
xin[1].append(im22/255.0)

pred_xshift = reg_cnn.predict(xin)[0][1]
pred_yshift = reg_cnn.predict(xin)[0][0]
print('pred xshift = ',pred_xshift)
print('pred yshift = ',pred_yshift)
#import pdb;pdb.set_trace()

pred_yshift = int(pred_yshift)
#import pdb;pdb.set_trace()
im2moved = np.zeros_like(im2)
abs_yshift = abs(pred_yshift)

if pred_yshift<0:
    im2moved[:-abs_yshift,:,:] = im2[abs_yshift:,:,:]
elif pred_yshift>0:
    im2moved[abs_yshift:,:,:] = im2[:-abs_yshift,:,:]
else:
    im2moved = im2

show_fused(im2moved,im2moved,win='im2moved',time=10)
show_fused(im1,im2,win='fused orig',time=30)
show_fused(im1,im2moved,win='fused moved',time=0)
#cv2.waitKey(0)
