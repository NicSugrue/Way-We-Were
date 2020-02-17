# Way-We-Were
Student AI Project at Emerson College

Hello!

Thank you very much for taking a look at my project. I am new to code, so this could get messy. But for now, I'm happy to try it out. Any and all help in the learning process would be greatly appreciated.

Take care,
Nic
-

In [4]:
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
In [5]:
from tensorflow.keras.preprocessing.image import load_img, img_to_array
In [6]:
model = VGG16()
In [7]:
model.summary()
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
In [8]:
import os
In [13]:
for file in os.listdir('sample'):
    print(file)
    full_path = 'sample/' + file
    
    image = load_img(full_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    y_pred = model.predict(image)
    label = decode_predictions(y_pred, top = 2)
    print(label)
    print()
bottle1.jpeg
[[('n04557648', 'water_bottle', 0.6603951), ('n04560804', 'water_jug', 0.08577988)]]

bottle2.jpeg
[[('n04557648', 'water_bottle', 0.5169559), ('n04560804', 'water_jug', 0.2630159)]]

bottle3.jpeg
[[('n04557648', 'water_bottle', 0.88239855), ('n04560804', 'water_jug', 0.051655706)]]

monitor.jpeg
[[('n03782006', 'monitor', 0.46309018), ('n03179701', 'desk', 0.16822667)]]

mouse.jpeg
[[('n03793489', 'mouse', 0.37214068), ('n03657121', 'lens_cap', 0.1903602)]]

mug.jpeg
[[('n03063599', 'coffee_mug', 0.46725288), ('n03950228', 'pitcher', 0.1496518)]]

pen.jpeg
[[('n02783161', 'ballpoint', 0.6506707), ('n04116512', 'rubber_eraser', 0.12477029)]]

wallet.jpeg
[[('n04026417', 'purse', 0.530347), ('n04548362', 'wallet', 0.24484588)]]

In [ ]:

In [ ]:
