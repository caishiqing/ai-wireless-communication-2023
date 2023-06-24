from tensorflow.keras import layers
import tensorflow as tf


def feedForward(res, embed_dim, hidden_dim):
    h = layers.Dense(hidden_dim, activation='relu')(res)
    x = layers.Dense(embed_dim)(h)
    x = layers.LayerNormalization()(x + res)
    return x


def preEncoding(num_tx=32, num_sc=128, num_layers=3, hidden_dims=None):
    embed_dim = 2 * num_tx
    if hidden_dims is None:
        hidden_dims = 2 * embed_dim

    x = layers.Input([3], name='coordinate')
    h = layers.Dense(embed_dim * num_sc)(x)
    h = layers.Reshape([num_sc, embed_dim])(h)
    for _ in range(num_layers):
        h = layers.MultiHeadAttention(8, embed_dim)(h, h)
        h = feedForward(h, embed_dim, hidden_dims)

    h = layers.Dense(hidden_dims, activation='relu')(h)
    h = layers.Dense(embed_dim)(h)
    h = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))(h)
    real, imag = tf.split(h, 2, axis=-1)
    y = tf.complex(real, imag, name='PreCoding')
    y = tf.expand_dims(y, -1)
    model = tf.keras.Model(x, y, name='preEncoder')
    return model


def trainModel(model, csi_shape, sigma=0.1):
    csi = layers.Input(shape=(csi_shape))
    loc, code = model.input, model.output
    code = tf.squeeze(code, -1)
    loss = snr_loss(csi, code)
    snr = snr_metric(sigma)(csi, code)
    train_model = tf.keras.Model([loc, csi], loss)
    train_model.add_loss(loss)
    train_model.add_metric(snr, name='snr')
    return train_model


def compute_gain(csi, code):
    # csi: (bc, rx, tx, sc, 2)
    # code: (bc, sc, tx)
    csi = tf.complex(csi[:, :, :, :, 0],  csi[:, :, :, :, 1])
    csi = tf.transpose(csi, [0, 3, 1, 2])
    code = tf.expand_dims(code, axis=2)
    r = tf.reduce_sum(csi * code, axis=-1)
    gain = tf.abs(tf.reduce_sum(tf.math.conj(r) * r, -1))
    return gain


def snr_loss(csi, code):
    gain = compute_gain(csi, code)
    snr = tf.math.log1p(gain)
    return -tf.reduce_mean(snr)


def snr_metric(sigma=0.1):
    def _metric(csi, code):
        gain = compute_gain(csi, code)
        snr = tf.math.log1p(gain/sigma) / tf.math.log(2.0)
        return tf.reduce_mean(snr)

    return _metric


tf.keras.utils.get_custom_objects().update({'tf': tf})
