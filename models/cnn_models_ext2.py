from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam

def model_1():
    model = Sequential()
    model.add(Conv2D(64, (5,5), input_shape= (40, 40, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def model_2():
    model = Sequential()
    model.add(Conv2D(64, (5,5), input_shape= (40, 40, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def model_3():
    model = Sequential()
    model.add(Conv2D(64, (5,5), input_shape= (40, 40, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    

    return model

def model_4():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(40, 40, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def model_5():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(40, 40, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def model_6():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(40, 40, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def model_7():
    model = Sequential()
    model.add(Conv2D(128, (3,3), input_shape= (40, 40, 3), activation='relu', ))
    model.add(Conv2D(128, (3,3), activation='relu', ))


    model.add(Conv2D(128, (3,3), activation='relu', ))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same' ))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3,3), activation='relu', padding='same' ))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same' ))
    
    model.add(Conv2D(32, (3,3), activation='relu', padding='same' ))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same' ))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same' ))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same' ))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def model_8():
    model = Sequential()
    model.add(Conv2D(128, (3,3), input_shape= (40, 40, 3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))


    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return model

def model_9():
    model = Sequential()
    model.add(Conv2D(512, (5,5), input_shape= (40, 40, 3), activation='relu'))
    model.add(Conv2D(512, (5,5), activation='relu'))


    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

