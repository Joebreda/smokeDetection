import os
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam


mobile = keras.applications.mobilenet.MobileNet()
def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# quickly inputting an image to test mobile net
#preprocessed_image = prepare_image('lab.jpg')
#predictions = mobile.predict(preprocessed_image)
#results = imagenet_utils.decode_predictions(predictions)
#print(results)

def create_model():
    #imports the mobilenet model and discards top layers
    base_model=MobileNet(weights='imagenet',include_top=False) 
    x=base_model.output
    # attach some new NN to learn on
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
    # specifiy inputs and outputs
    model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture    
    # freezes first 86 layers of mobileNet 
    for layer in model.layers[:87]:
        layer.trainable=False
    # print to check layer architecture and trainable layers
    for i,layer in enumerate(model.layers):
        print(i,layer.name, layer.trainable)
    # select optimizer, loss function, and metric
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

def train(model):
    # build training dataset
    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
    # this function is able to automatically distinguish classes based on number of folders and folder names
    train_generator=train_datagen.flow_from_directory('data/wildfire_smoke_data/train',
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)
    # use callback checkpoints to save model state at the end of each epoch
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    step_size_train=train_generator.n//train_generator.batch_size
    model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=10, callbacks=[cp_callback]) # add callbacks variable to training
    # Save the weights
    model.save_weights('checkpoints/trainedModel')
    return model    


def evaluate(model):
    # Evaluate model 
    test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
    # generate test data sequence
    test_generator=test_datagen.flow_from_directory('data/wildfire_smoke_data/validate',
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)
    step_size_test=test_generator.n//test_generator.batch_size
    results = model.evaluate_generator(generator=test_generator, steps=step_size_test)
    print('test loss, test acc:', results)
    return results


model = create_model()
model = train(model)
print('evaluate model post training')
results = evaluate(model)

# Create a basic model instance
model = create_model()
print('evaluate basic model that isnt trained')
evaluate(model)
# Loads the weights from manual save
model.load_weights('checkpoints/trainedModel')
print('evaluate model with loaded weights from final save after training')
evaluate(model)


#latest = tf.train.latest_checkpoint(checkpoint_dir)
# Load the previously saved weights
#model.load_weights(latest)


