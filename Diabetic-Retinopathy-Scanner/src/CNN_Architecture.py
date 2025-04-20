import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from dataExtraction import training_dataset, validation_dataset

# Define the ResNet-50 Architecture
class ResNet:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes


    def residual_block(self, x, output_dim, kernel_size=3, strides=1):
        identity = x
        
        # Conv Layer 1
        x = layers.Conv2D(output_dim, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Conv Layer 2
        x = layers.Conv2D(output_dim, kernel_size, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # If the dimensions of the input and output don't match, downsample
        input_dim = identity.shape[-1]
        if strides != 1 or output_dim != input_dim:
            identity = layers.Conv2D(output_dim, 1, strides=strides, padding='same')(identity)
            identity = layers.BatchNormalization()(identity)

        # # Add Squeeze-and-Excitation Attention Mechanism
        # x = self.se_block(x, reduction=16)

        # Add the identity to the output
        x = layers.add([x, identity])
        x = layers.ReLU()(x)

        return x

    def build_model(self):
        
        # Initialize the input layer of the CNN
        inputs = Input(self.input_shape)

        # First Conv Layer
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Residual Blocks with SE Attention
        x = self.create_layer(x, 64, 3)
        x = self.create_layer(x, 128, 4, strides=2)
        x = self.create_layer(x, 256, 6, strides=2)
        x = self.create_layer(x, 512, 3, strides=2)

        # Final Layer
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs, outputs)
        return model
    
    # Create layer using residual block
    def create_layer(self, x, output_dim, num_blocks, strides=1):
        x = self.residual_block(x, output_dim, strides=strides)
        for i in range(1, num_blocks):
            x = self.residual_block(x, output_dim)
        return x  
    
# Initialize the ResNet model
resnet = ResNet(input_shape=(224, 224, 3), num_classes=5)
resnet_model = resnet.build_model()

# Schedule the learning rate as the model trains
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  
    factor=0.8,          
    patience=5,   
    mode='auto',             
    verbose=1, 
    cooldown=5      
)

# Early stopping in case the model overfits
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,   
    mode='min',    
    verbose=1,      
    restore_best_weights=True
)

# Compile the model
resnet_model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Train the ResNet model
history = resnet_model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=[reduce_lr, early_stopping]
)

resnet_model.save("resNet_model.keras")
