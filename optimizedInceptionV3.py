from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inputV3
from keras.models import model_from_json
import keras.backend as K
import matplotlib.pyplot as plt

from kerassurgeon import Surgeon

import numpy as np
import json
import os
import math

def get_total_channels(model):
    start = None
    end = None
    channels = 0
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            channels += layer.filters
    return channels

def get_model_apoz(model, generator):
    # Get APoZ
    start = None
    end = None
    apoz = []
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            print(layer.name)
            apoz.extend([(layer.name, i, value) for (i, value)
                         in enumerate(get_apoz(model, layer, generator))])

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    return apoz_df

popped_layers_list = []
percent_pruning = 2
output_dir = 'inception_faces/'
validation_data_dir = output_dir+'data/validation/'

# Set up data generators
test_datagen = ImageDataGenerator(
        preprocessing_function=inception_v3.preprocess_input)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(299, 299),
    batch_size=val_batch_size,
    class_mode='categorical')

modelInceptionV3 = InceptionV3(weights='imagenet', include_top=False)
modelInceptionV3.summary()
print(len(modelInceptionV3.layers))
print("")
print("**************************************************************************************************NEW MODEL******************************************************************************")
print("")
#popped_layer = modelInceptionV3.layers.pop()
#modelInceptionV3.summary()
#json_string = modelInceptionV3.to_json()

'''''with open('optimized_model.json', 'w') as outfile:  
    json.dump(json.loads(json_string), outfile)
print(len(modelInceptionV3.layers))'''''

"""new_model_json = open('optimized_model.json').read()
print("Loading model.................")
loaded_model = model_from_json(new_model_json)
print("Model loaded.")
loaded_model.summary()"""

total_channels = get_total_channels(modelInceptionV3)
print("Total number of channels: "+str(total_channels))

n_channels_delete = int(math.floor(percent_pruning / 100 * total_channels))
print("Number of channels to delete: "+str(n_channels_delete))

percent_pruned = 0
# If percent_pruned > 0, continue pruning from previous checkpoint
if percent_pruned > 0:
    checkpoint_name = ('inception_flowers_pruning_' + str(percent_pruned)+ 'percent')

while percent_pruned <= total_percent_pruning:
        # Prune the model
        apoz_df = get_model_apoz(model, validation_generator)
        percent_pruned += percent_pruning
        print('pruning up to ', str(percent_pruned),
              '% of the original model weights')
        model = prune_model(model, apoz_df, n_channels_delete)

        # Clean up tensorflow session after pruning and re-load model
        checkpoint_name = ('inception_pruning_' + str(percent_pruned)+ 'percent')
        model.save(output_dir + checkpoint_name + '.h5')