import tensorflow as tf
import bpy
from mathutils import Vector

class bpy_Layer(object):
    def __init__(self, config: dict) -> object:
        self.config = config['config']
        self.classname = config['class_name']
        self.weights = []

    def show(self, offset: int=0, spacing: int=4) -> None:
        print(self.shape)
        x ,y = self.shape[1:3] or (1,1)
        spacing = 4

        center = ((x+1)*spacing/2, (y+1)*spacing/2)

        bpy.ops.mesh.primitive_ico_sphere_add()
        orig_neuron = bpy.context.active_object
        
        
        for x in range(1, x+1):
            for y in range(1, y+1):
                m = orig_neuron.data.copy()
                o = bpy.data.objects.new(f'neuron', m)
                o.location = Vector(((x * spacing)-center[0], (y * spacing)-center[1],offset))
                bpy.context.collection.objects.link(o)

        bpy.ops.object.delete()

       
class bpy_InputLayer(bpy_Layer):
    def __init__(self, config):
        super().__init__(config)
        self.shape = (*self.config['batch_input_shape'], 1)

class bpy_Normalization(bpy_Layer):
    def __init__(self, config):
        super().__init__(config)
        self.shape = (*self.config['batch_input_shape'], 1)

class bpy_Dense(bpy_Layer):
    def __init__(self, config):
        super().__init__(config)
        self.units = self.config['units']
        self.shape = (None, self.units, 1)
        self.activation = self.config['activation']

class bpy_Conv2D(bpy_Layer):
    def __init__(self, config):
        super().__init__(config)
        self.activation = self.config['activation']
        self.kernel_size = self.config['kernel_size']
        self.filters = self.config['filters']
        self.strides = self.config['strides']

class bpy_MaxPooling2D(bpy_Layer):
    def __init__(self, config):
        super().__init__(config)
        self.pool_size = self.config['pool_size']
        self.strides = self.config['strides']

class bpy_Flatten(bpy_Layer):
    def __init__(self, config):
        super().__init__(config)
        
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
    
    def show(self, spacing: int=10) -> None:

        for i in range(len(self.layers)):
            collection = bpy.data.collections.new(name=f'Layer {i}')
            bpy.context.scene.collection.children.link(collection)

            # set collection active
            bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[i]

            try:
                self.layers[i].show(offset=i*spacing)
            except Exception as e:
                print(e)

    @staticmethod
    def clear():
        for i in bpy.data.collections: bpy.data.collections.remove(i)
        bpy.data.orphans_purge() 

   