
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import visualkeras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

def base_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape= (20, 20, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(8, (3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = base_model()

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = '#0CF080'
color_map[MaxPooling2D]['fill'] = '#F0260C'
color_map[Flatten]['fill'] = '#32463C'
color_map[Dense]['fill'] = '#B04335'
color_map[Dropout]['fill'] = '#3B9B6C'
visualkeras.layered_view(model, to_file='modelo.png', 
                         color_map=color_map, legend=True)
visualkeras.layered_view(model, to_file='modelo.png', color_map=color_map)

