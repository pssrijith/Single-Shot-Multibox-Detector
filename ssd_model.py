from transfer_learn import VGG16

from keras.layers import Flatten, merge,Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.models import Model

def build(input_shape=(300,300,3), num_classes=2, num_def_boxes=4, pre_train_net = "imagenet") :
    
    # create the vgg16 object from which we can get the basenet for our SSD
    vgg16 = VGG16.VGG16(input_shape=input_shape, pre_train_net= pre_train_net , include_top=False)
    
    basenet_model = vgg16.get_basenet_model(trunc_at_layer='block5_conv3')
    
    block5_conv3_layer = basenet_model.layers[len(basenet_model.layers) -1]
    block5_conv3_tensor = block5_conv3_layer.output 
    
    fc6 = Conv2D(1024,3,3, border_mode='same', activation='relu', name='fc6')(block5_conv3_tensor)
    
    fc7 = Conv2D(1024,1,1, border_mode='same', activation='relu', name='fc7')(fc6)
    
    conv_8_1 = Conv2D(256,1,1, border_mode='same', activation='relu', name='conv_8_1')(fc7)
    conv_8_2 = Conv2D(512,3,3, subsample=(2, 2), border_mode='same', activation='relu', name='conv_8_2')(conv_8_1)
    
    conv_9_1 = Conv2D(128,1,1, border_mode='same', activation='relu', name='conv_9_1')(conv_8_2)
    conv_9_2 = Conv2D(256,3,3, subsample=(2, 2), border_mode='same', activation='relu', name='conv_9_2')(conv_9_1)
    
    conv_10_1 = Conv2D(128,1,1, border_mode='same', activation='relu', name='conv_10_1')(conv_9_2)
    conv_10_2 = Conv2D(256,3,3, subsample=(2, 2), border_mode='same', activation='relu', name='conv_10_2')(conv_10_1)
    
    conv_11_1 = Conv2D(128,1,1, border_mode='same', activation='relu', name='conv_11_1')(conv_10_2)
    conv_11_2 = Conv2D(256,3,3, subsample=(2, 2), border_mode='same', activation='relu', name='conv_11_2')(conv_11_1)    
    
    # Detection convolutions at different scales
    
    # detect from block4_conv3 from the base net model
    i,block4_conv3_lyr = vgg16.getLayerByName(model=basenet_model, layer_name='block4_conv3')
    det_classes_block4_conv3 = Conv2D(num_def_boxes * num_classes, 3,3,
                             border_mode ='same', activation='relu', name='det_classes_block4_conv3')(block4_conv3_lyr.output)
    det_class_block4_conv3_flat = Flatten(name='det_classes_block4_conv3_flat')(det_classes_block4_conv3)
    
    det_loc_block4_conv3 = Conv2D(num_def_boxes * 4, 3,3,
                               border_mode ='same', activation='relu', name='det_loc_block4_conv3')(block4_conv3_lyr.output)
    det_loc_block4_conv3_flat = Flatten(name='det_loc_block4_conv3_flat')(det_loc_block4_conv3)
    
    # detect from fc7
    det_classes_fc7 = Conv2D(num_def_boxes * num_classes, 3,3,
                               border_mode ='same', activation='relu', name='det_classes_fc7')(fc7)
    det_class_fc7_flat = Flatten(name='det_classes_fc7_flat')(det_classes_fc7)
    det_loc_fc7 = Conv2D(num_def_boxes * 4, 3,3,
                               border_mode ='same', activation='relu', name='det_loc_fc7')(fc7)    
    det_loc_fc7_flat = Flatten(name='det_loc_fc7_flat')(det_loc_fc7)
    
    # detect from conv_8_2
    det_classes_conv_8_2 = Conv2D(num_def_boxes * num_classes, 3,3,
                               border_mode ='same', activation='relu', name='det_classes_conv_8_2')(conv_8_2)
    det_class_conv_8_2_flat = Flatten(name='det_classes_conv_8_2_flat')(det_classes_conv_8_2)
    det_loc_conv_8_2 = Conv2D(num_def_boxes * 4, 3,3,
                               border_mode ='same', activation='relu', name='det_loc_conv_8_2')(conv_8_2)    
    det_loc_conv_8_2_flat = Flatten(name='det_loc_conv_8_2_flat')(det_loc_conv_8_2)
    
    # detect from conv_9_2
    det_classes_conv_9_2 = Conv2D(num_def_boxes * num_classes, 3,3,
                               border_mode ='same', activation='relu', name='det_classes_conv_9_2')(conv_9_2)
    det_class_conv_9_2_flat = Flatten(name='det_classes_conv_9_2_flat')(det_classes_conv_9_2)
    det_loc_conv_9_2 = Conv2D(num_def_boxes * 4, 3,3,
                               border_mode ='same', activation='relu', name='det_loc_conv_9_2')(conv_9_2)    
    det_loc_conv_9_2_flat = Flatten(name='det_loc_conv_9_2_flat')(det_loc_conv_9_2)
    
    
    # detect from conv_10_2
    det_classes_conv_10_2 = Conv2D(num_def_boxes * num_classes, 3,3,
                               border_mode ='same', activation='relu', name='det_classes_conv_10_2')(conv_10_2)
    det_class_conv_10_2_flat = Flatten(name='det_classes_conv_10_2_flat')(det_classes_conv_10_2)
    det_loc_conv_10_2 = Conv2D(num_def_boxes * 4, 3,3,
                               border_mode ='same', activation='relu', name='det_loc_conv_10_2')(conv_10_2)    
    det_loc_conv_10_2_flat = Flatten(name='det_loc_conv_10_2_flat')(det_loc_conv_10_2)
    input_tensor = basenet_model.layers[0].input
    
    # detect from conv_11_2
    det_classes_conv_11_2 = Conv2D(num_def_boxes * num_classes, 3,3,
                               border_mode ='same', activation='relu', name='det_classes_conv_11_2')(conv_11_2)
    det_class_conv_11_2_flat = Flatten(name='det_classes_conv_11_2_flat')(det_classes_conv_11_2)
    det_loc_conv_11_2 = Conv2D(num_def_boxes * 4, 3,3,
                               border_mode ='same', activation='relu', name='det_loc_conv_11_2')(conv_11_2)    
    det_loc_conv_11_2_flat = Flatten(name='det_loc_conv_11_2_flat')(det_loc_conv_11_2)
    
    
    # final detections layer
    
    pred_y_classes = merge([det_class_block4_conv3_flat,
                            det_class_fc7_flat,
                            det_class_conv_8_2_flat,
                            det_class_conv_9_2_flat,
                            det_class_conv_10_2_flat,
                            det_class_conv_11_2_flat], mode='concat', concat_axis=1, name='pred_y_classes')
    pred_y_loc = merge([det_loc_block4_conv3_flat,
                            det_loc_fc7_flat,
                            det_loc_conv_8_2_flat,
                            det_loc_conv_9_2_flat,
                            det_loc_conv_10_2_flat,
                            det_loc_conv_11_2_flat], mode='concat', concat_axis=1, name='pred_y_loc')
    
    detection_layer_tensor = merge([pred_y_classes,pred_y_loc],mode='concat', concat_axis=1, name='detection_layer')
    
    input_tensor = basenet_model.layers[0].input
    
    return Model(input_tensor, detection_layer_tensor)