import tensorflow as tf
import keras
import cv2
import numpy as np

def show_fused(im1,im2,win='fused'):
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    fused = np.zeros_like(im1)
    fused[:,:,1] = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    fused[:,:,2] = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    cv2.imshow(win, fused)



checkpoint_path = 'weights/'
reg_cnn = tf.keras.models.load_model(checkpoint_path)

im1 = cv2.imread('samples/1903_.png',1)
im2 = cv2.imread('samples/1903_-13.png',1)

xin=[[],[]]
im11=np.expand_dims(im1,0)
im22=np.expand_dims(im2,0)
xin[0].append(im11/255.0)
xin[1].append(im22/255.0)

pred_yshift = reg_cnn.predict(xin)[0][0]
print('pred yshift = ',pred_yshift)

pred_yshift = int(pred_yshift)
#import pdb;pdb.set_trace()
im2moved = np.zeros_like(im2)
if pred_yshift>=0:
    im2moved[:,:-pred_yshift,:] = im2[:,pred_yshift:,:]
else:
    im2moved[:,pred_yshift:,:] = im2[:,:-pred_yshift,:]

show_fused(im1,im2,win='fused orig')
show_fused(im1,im2moved,win='fused moved')
cv2.waitKey(0)

#import pdb;pdb.set_trace()