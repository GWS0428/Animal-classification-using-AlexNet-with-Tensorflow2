"""Define the model."""

import tensorflow as tf

def build_model(params, input_shape = (227, 227, 3), classes = 10):
    """Compute logits of the model (output distribution)
    Args:
        params: (Params) hyperparameters
        input_shape: (tuple) shape of a input image
        classes: (int) number of classes of image
    Returns:
        model: (tf.keras.Model) compiled model
    """
    X_input = tf.keras.Input(input_shape)
    X = X_input
    X = tf.keras.layers.Conv2D(96, (11, 11), strides = (4, 4), activation = "relu", name = 'conv1')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = tf.keras.layers.Conv2D(256, (5, 5), padding = "same",activation = "relu", name = 'conv2')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = tf.keras.layers.Conv2D(256, (3, 3), padding = "same",activation = "relu", name = 'conv5')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(1))(X)
    X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(2))(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)
       
    # Create model
    model = tf.keras.Model(inputs = X_input, outputs = X, name='ALEXNET')
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model