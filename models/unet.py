import numpy as np 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

def unet_model(input_size=(128,128,3)):
    inputs = Input(input_size)

    #encoder
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    #bridge
    conv5 = Conv2D(1024, (3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3,3), activation='relu', padding='same')(conv5)

    #Decoder
    up6 = Conv2D(512, (2,2), activation='relu', padding='same')(UpSampling2D(size=(2,2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, (3,3), activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, (3,3), activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, (2,2), activation='relu', padding='same')(UpSampling2D(size=(2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, (3,3), activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, (3,3), activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, (2,2), activation='relu', padding='same')(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, (3,3), activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, (3,3), activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, (2,2), activation='relu', padding='same')(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, (3,3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(2, (1,1), activation='softmax')(conv9) 

    model = Model(inputs=inputs, outputs=[outputs])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model




