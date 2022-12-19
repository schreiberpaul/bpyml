import tensorflow as tf
import pprint as pp
from bpy_layers import *

class bpy_model():
    def __init__(self, model: tf.keras.Model) -> object:
        self.name = ''
        self.layers = []
        self.model = model

        self.build()
    
    @classmethod
    def from_file(cls, path: str):
        try:
            m = tf.keras.models.load_model(path)
        except Exception as e:
            print(e)
            exit()

        return cls(m)
    
    def build(self) -> None:
        
        config = self.model.get_config()
        self.name = config['name']

        for layer in config['layers']:
        
            match layer['class_name']:
                case 'InputLayer':
                    self.layers.append(bpy_InputLayer(layer))
                
                case 'Normalization':
                    self.layers.append(bpy_Normalization(layer))

                case 'Dense':
                    self.layers.append(bpy_Dense(layer))

                case 'Conv2D':
                    self.layers.append(bpy_Conv2D(layer))
                
                case 'MaxPooling2D':
                    self.layers.append(bpy_MaxPooling2D(layer))
                
                case 'Flatten':
                    self.layers.append(bpy_Flatten(layer))
                
                case _:
                    raise NotImplementedError

    
if __name__ == '__main__':
    # path = 'test_models/ps2_model.h5'
    path = 'test_models/convolution_simpel'

    model = bpy_model.from_file(path)

    i=0
    for layer in model.layers:
        layer.show(offset=i)
        i += 10
