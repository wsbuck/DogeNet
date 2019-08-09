from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from PIL import Image

import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import os
import json

def main():
    data_path = "../data/Images/"
    n_classes = len(os.listdir(data_path))
    
    base_model = MobileNet(weights="imagenet", include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    
    for layer in model.layers:
        layer.trainable = False
    
    for layer in model.layers[:20]:
        layer.trainable = False
    
    for layer in model.layers[20:]:
        layer.trainable = True
    
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow_from_directory('../data/Images',
                                                      target_size=(224,224),
                                                      color_mode='rgb',
                                                      batch_size=32,
                                                      class_mode='categorical',
                                                      shuffle=True)
    
    model.compile(
        optimizer='Adam', loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    step_size_train = train_generator.n // train_generator.batch_size

    #filepath = "./saved/saved-model-{epoch:02d}-{acc:.2f}.hdf5"
    filepath = "./saved/saved-model-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='acc',
        verbose=1,
        save_best_only=False, mode='max'
    )
    callbacks_list = [checkpoint]

    model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train,
                    epochs=200, callbacks=[checkpoint])
    model.save('./saved/model_v1_final.h5')

    tfjs.converters.save_keras_model(model, "./")

    outputClasses = {}
    temp = json.dumps(train_generator.class_indices)
    temp = json.loads(temp)
    
    for k, v in temp.items():
        if k == "Nothing":
            outputClasses[v] = k
        else:
            key = k[10:].replace("_", " ").title()
            outputClasses[v] = key

    
    with open('./outputClasses.json', 'w') as outfile:
        json.dump(outputClasses, outfile)

if __name__ == '__main__':
    main()
