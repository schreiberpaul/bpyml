import tensorflow as tf
import pprint as pp
import bpy
from mathutils import Vector

class bpy_Layer(object):
    def __init__(self, config: dict) -> object:
        self.config = config['config']
        self.classname = config['class_name']
        self.weights = []

    def show(self, offset: int=0, spacing: int=4) -> None:
    
        dims = self.shape[1:2]
        x ,y = dims
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
        self.shape = self.config['batch_input_shape']

class bpy_Normalization(bpy_Layer):
    def __init__(self, config):
        super().__init__(config)
        self.shape = self.config['batch_input_shape']

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
        

if __name__ == '__main__':
    
    m = tf.keras.models.load_model('test_models/ps2_model.h5')
    config = m.get_config()['layers']

    pp.pp(config)
   
   