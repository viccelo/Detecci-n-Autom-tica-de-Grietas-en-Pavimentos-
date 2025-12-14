import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose,
    concatenate, BatchNormalization
)

def conv_block(inputs, num_filters, use_bn=True, dropout_rate=0.0):
    """
    Bloque de convolución doble con BatchNorm opcional y Dropout.
    """
    x = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    if use_bn:
        x = BatchNormalization()(x)
    x = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def encoder_block(inputs, num_filters, use_bn=True, dropout_rate=0.0):
    x = conv_block(inputs, num_filters, use_bn, dropout_rate)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters, use_bn=True, dropout_rate=0.0):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = concatenate([x, skip_features])
    x = conv_block(x, num_filters, use_bn, dropout_rate)
    return x

def crear_modelo_unet(input_size=(256, 256, 3), base_filters=32, use_bn=True, dropout_rate=0.3):
    """
    U-Net optimizada con menos filtros, Dropout y BatchNorm opcional.
    """
    inputs = Input(input_size)

    # --- Encoder ---
    s1, p1 = encoder_block(inputs, base_filters, use_bn, dropout_rate/2)
    s2, p2 = encoder_block(p1, base_filters*2, use_bn, dropout_rate/2)
    s3, p3 = encoder_block(p2, base_filters*4, use_bn, dropout_rate/2)
    s4, p4 = encoder_block(p3, base_filters*8, use_bn, dropout_rate)

    # --- Bottleneck ---
    b1 = conv_block(p4, base_filters*16, use_bn, dropout_rate)

    # --- Decoder ---
    d1 = decoder_block(b1, s4, base_filters*8, use_bn, dropout_rate)
    d2 = decoder_block(d1, s3, base_filters*4, use_bn, dropout_rate/2)
    d3 = decoder_block(d2, s2, base_filters*2, use_bn, dropout_rate/2)
    d4 = decoder_block(d3, s1, base_filters, use_bn, dropout_rate/2)

    # --- Output ---
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)

    modelo = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    print("✅ Modelo U-Net (con Dropout y optimizada) creado exitosamente.")
    return modelo

if __name__ == "__main__":
    modelo = crear_modelo_unet()
    modelo.summary()
