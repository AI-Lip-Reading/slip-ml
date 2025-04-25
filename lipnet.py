import tensorflow as tf
from tensorflow.keras.layers import (
    Conv3D, ZeroPadding3D, MaxPooling3D, Dense, 
    Activation, Dropout, Flatten, Bidirectional, 
    TimeDistributed, GRU, Input
)
from tensorflow.keras.models import Model
from lipnet.core.layers import CTC

class LipNet:
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, absolute_max_string_len=32, output_size=28):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.build()

    def build(self):
        # Use channels_last format (TensorFlow default)
        input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)
        
        self.input_data = Input(name='the_input', shape=input_shape, dtype=tf.float32)
        
        # First 3D Convolutional Block
        x = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        x = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), activation='relu', 
                  kernel_initializer='he_normal', name='conv1')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(x)
        x = Dropout(0.5)(x)

        # Second 3D Convolutional Block
        x = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(x)
        x = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), activation='relu',
                  kernel_initializer='he_normal', name='conv2')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(x)
        x = Dropout(0.5)(x)

        # Third 3D Convolutional Block
        x = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(x)
        x = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), activation='relu',
                  kernel_initializer='he_normal', name='conv3')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(x)
        x = Dropout(0.5)(x)

        # Reshape for RNN
        x = TimeDistributed(Flatten())(x)

        # Bidirectional GRU layers
        x = Bidirectional(GRU(256, return_sequences=True, 
                            kernel_initializer='orthogonal', name='gru1'))(x)
        x = Bidirectional(GRU(256, return_sequences=True,
                            kernel_initializer='orthogonal', name='gru2'))(x)

        # Dense layer for character predictions
        x = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(x)
        self.y_pred = Activation('softmax', name='softmax')(x)

        # Input layers for CTC loss
        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype=tf.float32)
        self.input_length = Input(name='input_length', shape=[1], dtype=tf.int64)
        self.label_length = Input(name='label_length', shape=[1], dtype=tf.int64)

        # CTC loss layer
        self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        # Create model
        self.model = Model(
            inputs=[self.input_data, self.labels, self.input_length, self.label_length],
            outputs=self.loss_out
        )

    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()

    def predict(self, input_batch):
        return self.test_function([input_batch, 0])[0]

    @property
    def test_function(self):
        return tf.keras.backend.function(
            [self.input_data, tf.keras.backend.learning_phase()],
            [self.y_pred, tf.keras.backend.learning_phase()]
        )