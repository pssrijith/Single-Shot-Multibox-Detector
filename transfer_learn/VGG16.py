from keras import applications
from keras.models import Model


'''

Utility class to get a pre-initialized VGG16 model fro transfer learning. The class provides method to get the 
full VGG16 model or a truncated base net model upto the layer that you are interested in.

'''

class VGG16() :
    '''
    Constructor to create the VGG16 object. It stores all necessary input parameters that will be then used by subsequent 
    methods to create a VGG16 model, initialized with weights from the source defined in weights_from parameter.
    
    Arguments:
        input_shape (tuple): The input image shape e.g. input_shape=(300,300,3) for a 300x300 3-channel image
        pre_train_net(str)  : allowed values('imagenet'). This refers to the source of the trained weights. 
                             Currently we only support the imagenet trained weights. 
        include_top(boolean): If this is false, we will leave out the FC layers from the model
    '''
    def __init__(self, input_shape=None,pre_train_net="imagenet", include_top=False) :
        
        if input_shape is None or len(input_shape) != 3:
            raise ValueError('Invaid input shape. Please provide a valid tuple. e.g., input_shape=(255,255,3)')
        self.input_shape = input_shape
        self.pre_train_net = pre_train_net
        self.include_top = include_top
        
    '''
    get_basenet_model
    This is the method that you will need to call to get a new VGG16 model initialized with weights 
    and truncated upto the layer that you need 
    Note: This method  will get layers from  a new VGG16 model everytime you call it so you get a fresh model everytime. 
    
    Arguments:
        trunc_at_layer (str) : the name of the layer until which you need. layers after this layer will 
                                  be truncated. layer names should be one of the following
                                    'input_1'
                                    'block1_conv1'
                                    'block1_conv2'
                                    'block1_pool'
                                    'block2_conv1'
                                    'block2_conv2'
                                    'block2_pool'
                                    'block3_conv1'
                                    'block3_conv2'
                                    'block3_conv3'
                                    'block3_pool'
                                    'block4_conv1'
                                    'block4_conv2'
                                    'block4_conv3'   -- ssd paper - pre-train layers upto this vgg layer
                                    'block4_pool'
                                    'block5_conv1'
                                    'block5_conv2'
                                    'block5_conv3'
                                    'block5_pool'

        freeze_all_weights(bool): To freeze all weights in the layers(i.e. layers will not be affected by 
                                  backpropagation)
                                  
    Return: dictionary of keras layers (key: layer_name).
       
    '''
    def get_basenet_model(self, trunc_at_layer='block5_conv3', freeze_all_weights = True) :
        
        # start with a new VGG16 pretrained model 
        full_model = applications.VGG16(weights = self.pre_train_net, 
                                        include_top=self.include_top, input_shape=self.input_shape)
        # find the layer index for the trunc_at_layer
        index, trunc_at_layer = self.getLayerByName(full_model, trunc_at_layer)
        
        # note : output attr for a layer gives tensor for the layer - need for Model constructor
        input_layer_tensor = full_model.layers[0].output 
        trunc_at_layer_tensor = trunc_at_layer.output
        
        # build the truncated model
        baseNetModel = Model(input_layer_tensor, trunc_at_layer_tensor)
        
        if freeze_all_weights is True :
            # freeze weights in all base net model layers
            for layer in baseNetModel.layers :
                layer.trainable = False
                
        return baseNetModel
    
    '''
    getLayerByName - utility method to get a layer in a Model object by layer name
    '''
    def getLayerByName(self, model, layer_name) :
        if model is None:
           raise ValueError("Missing full VGG model in arguments to method")
           
        for i,layer in enumerate(model.layers) :
            if layer.name == layer_name :
                print ('Found layer : ', layer_name, ' at index ', i)
                return i,layer 
        raise ValueError("Invalid layer name - " + layer_name)
    