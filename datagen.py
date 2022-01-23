import os
import tensorflow as tf
import cv2
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    'Initialisation'
    def __init__(self, path='train', batch_size=8, outdim=(128,128,3),show=False, write=False):
        self.data_path =path
        self.show = show
        self.write = write
        self.batch_size = batch_size
        self.outdim = outdim
        self.x_shift_max = 1
        self.y_shift_max = 20
        self.curr_epoch=0
        self.xdata = self.load_data()
        
        self.shuffle = True
        self.on_epoch_end()
        self.img_indx=0
        

    def __len__(self):
        'num batches per epoch'
        return len(self.xdata) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch data'
        list_indx = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X,y = self.get_batchdata(list_indx)
        return X, y
    
    def on_epoch_end(self):
        self.curr_epoch+=1
        if self.curr_epoch%10==0:
            self.load_data()
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    
    def shift_aug(self, img):
        'Apply shift aug and return shifted images and x y shifts'
        
        h,w = img.shape[:2]
        x1 = np.random.randint(0,self.x_shift_max)
        x2 = np.random.randint(0,self.x_shift_max)
        y1 = np.random.randint(0,self.y_shift_max)
        y2 = np.random.randint(0,self.y_shift_max)
        
        img1 = img[y1:y1+self.outdim[0], x1:x1+self.outdim[1],:]
        img2 = img[y2:y2+self.outdim[0], x2:x2+self.outdim[1],:]
        xshift = x2-x1
        yshift = y2-y1

        if img1.shape[0]!=128 or img1.shape[1]!=128 or img2.shape[0]!=128 or img2.shape[1]!=128:
            print('img1shape',img1.shape)
            print('img2shape',img2.shape)
            #import pdb;pdb.set_trace() 

        return img1, img2, xshift, yshift

    def get_batchdata(self, list_indx):
        'Generate batchsize samples Eg: X: (nsamples, *dim, nsamples'
        #X = np.empty(())
        batch_x = []
        batch_y = []
        batch_x.append(np.zeros((len(list_indx), *self.outdim)))
        batch_x.append(np.zeros((len(list_indx), *self.outdim)))
        
        batch_y.append(np.zeros((len(list_indx))))
        batch_y.append(np.zeros((len(list_indx))))

        for i in range(len(list_indx)):
            img = self.xdata[list_indx[i]]
            im1,im2,xshift,yshift = self.shift_aug(img)

            if self.show:
                print(f'xshift{xshift},yshift{yshift}')
                
                cv2.namedWindow('im1',cv2.WINDOW_NORMAL)
                cv2.namedWindow('im2',cv2.WINDOW_NORMAL)
                cv2.imshow('im1',im1)
                cv2.imshow('im2',im2)
                cv2.waitKey(0)
            if self.write:
                nam = 'samples/'+str(self.img_indx) +"_"
                self.img_indx+=1
                cv2.imwrite(nam+'.png', im1)
                cv2.imwrite(nam+'x_'+str(xshift)+'__y_'+str(yshift)+'.png', im2)

            batch_x[0][i] = im1/255.0
            batch_x[1][i] = im2/255.0

            
            batch_y[0][i] = xshift
            batch_y[1][i] = yshift #/self.y_shift_max
        return batch_x, batch_y

    def load_data(self):
        print('Loading data...')
        xdata=[]
        h1,w1 = self.outdim[:2]
        hmin = h1+self.y_shift_max
        wmin = w1+self.x_shift_max
        for root, dir, files in os.walk(self.data_path):
            for file_ in files:
               
                img = cv2.imread(os.path.join(root,file_),1)
                if img.shape[0]>hmin and img.shape[1]>wmin:
                    xdata.append(img)

        self.indexes = np.arange(len(xdata))
        return xdata



if __name__ == '__main__':
    # wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O cats_and_dogs_filtered.zip
    # unzip cats_and_dogs_filtered -d cat_dog
    datagen = DataGenerator(path = 'cat_dog/cats_and_dogs_filtered/validation',show=True,write=True)
    batch_x, batch_y  = datagen.__getitem__(0)
    #import pdb;pdb.set_trace()
    print('batchx[0]',batch_x[0].shape)
    print('batchy len',len(batch_y))
    print('batch_y',batch_y)
    
    import pdb;pdb.set_trace()
    





