import tensorflow as tf

class bpy_model():
    def __init__(self) -> None:
        self.name = ''
        self.layers = []
    
    def load(self, path) -> None:
        try:
            m = tf.keras.models.load_model(path)
        except Exception as e:
            print(e)
            exit()
        
        config = m.get_config()
        for layer in config['layers']:
            dict = {'type' : layer['class_name']}

            if 'activation' in layer['config']:
                dict.update({'activation' : layer['config']['activation']}) 

            if 'units' in layer['config']:
                dict.update({'neurons' : layer['config']['units']})
          
            if 'batch_input_shape' in layer['config']:
                dict.update({'input_shape' : layer['config']['batch_input_shape']})
         
            self.layers.append(dict)

if __name__ == '__main__':
    path = 'ps2_model.h5'

    model = bpy_model()
    model.load(path=path)

    for layer in model.layers:
        print('')
        for key, value in layer.items():
            print(f'{key}: {value}')