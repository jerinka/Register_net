import tensorflow as tf
import cv2
#from tensorflow import keras
import keras
from keras import layers
from datagen import DataGenerator
from keras.applications import resnet_v2

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# your code


def get_base_Resnet(input_shape=(128,128,3)):
    model1 =  resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    layer_name = 'conv2_block1_2_relu'
    model2= keras.Model(inputs=model1.input, outputs=model1.get_layer(layer_name).output)   
    return model2  

def get_base_model(input_shape=(128,128,3)):
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(64,5,activation='relu',strides=2,padding='valid')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,5,activation='relu',strides=2,padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256,5,activation='relu',strides=2,padding='valid')(x)
    output = layers.BatchNormalization()(x)
    return keras.Model(input, output,name="base_model")

def get_model(input_shape=(128,128,3)):
    input1 = layers.Input(shape=input_shape)
    input2 = layers.Input(shape=input_shape)
    basemodel = get_base_Resnet() #get_base_model()
    basemodel.trainable = False
    basemodel.summary()
    feat1 = basemodel(input1)
    feat2 = basemodel(input2)

    x = layers.Concatenate(axis=-1)([feat1, feat2])
    x = layers.Conv2D(256, 3,strides=1,padding='valid',activation='relu')(x)
    x = layers.Conv2D(256, 3,strides=1,padding='valid',activation='relu')(x)
    x = layers.GlobalMaxPooling2D()(x)

    #x = layers.Flatten()(x)
    x = layers.Dense(128,activation='sigmoid')(x)
    x = layers.Dense(128,activation='sigmoid')(x)
    output = layers.Dense(1, activation='linear')(x)
    return keras.Model([input1, input2], output)

if __name__=='__main__':
    input_shape=(128,128,3)
    base=get_base_model()
    base.summary()
    model = get_model()
    model.summary()
    if 1:
        checkpoint_path = 'weights/'
        model = tf.keras.models.load_model(checkpoint_path)

    #model.build(input_shape)
    datagen = DataGenerator(path = 'cat_dog/cats_and_dogs_filtered/train')
    batch_x, batch_y  = datagen.__getitem__(0)
    print('batchx[0]',batch_x[0].shape)
    print('actual batchy:',batch_y)
    print('predicted y:',model.predict(batch_x))
    import pdb;pdb.set_trace()
    

