import tensorflow as tf
import scipy.io as scio
from ModelDesign import preEncoding
import numpy as np
print(tf.__version__)

Rx_num = 4
Tx_num = 32
SC_num = 128


def EqChannelGain(channel, PrecodingVector):
    # The authentic CSI
    HH = np.reshape(channel, (-1, Rx_num, Tx_num, SC_num, 2))  # Rx, Tx, Subcarrier, RealImag
    HH_complex = HH[:, :, :, :, 0] + 1j * HH[:, :, :, :, 1]  # Rx, Tx, Subcarrier
    HH_complex = np.transpose(HH_complex, [0, 3, 1, 2])

    # Power Normalization of the precoding vector
    Power = np.matmul(np.transpose(np.conj(PrecodingVector), (0, 1, 3, 2)), PrecodingVector)
    PrecodingVector = PrecodingVector / np.sqrt(Power)

    # Effective channel gain
    R = np.matmul(HH_complex, PrecodingVector)
    R_conj = np.transpose(np.conj(R), (0, 1, 3, 2))
    h_sub_gain = np.matmul(R_conj, R)
    h_sub_gain = np.reshape(np.absolute(h_sub_gain), (-1, SC_num))  # channel gain of SC_num subcarriers
    return h_sub_gain


def DataRate(h_sub_gain, sigma2_UE):  # Score
    SNR = h_sub_gain / sigma2_UE
    Rate = np.log2(1 + SNR)  # rate
    Rate_OFDM = np.mean(Rate, axis=-1)  # averaging over subcarriers
    Rate_OFDM_mean = np.mean(Rate_OFDM)  # averaging over CSI samples
    return Rate_OFDM_mean


# Parameters
sigma2_UE = 0.1

# Load data
f = scio.loadmat('./data/train.mat')
data = f['train']
location_data = data[0, 0]['loc']
channel_data = data[0, 0]['CSI']


# ### Load model
RadioMap_address = 'model.h5'
RadioMap = tf.keras.models.load_model(RadioMap_address)

# Prediction
PrecodingVector = RadioMap.predict(location_data)
#PrecodingVector = np.reshape(PrecodingVector, (-1, SC_NUM, Tx_num, 1))

# Calculate the score
SubCH_gain_codeword = EqChannelGain(channel_data, PrecodingVector)
data_rate = DataRate(SubCH_gain_codeword, sigma2_UE)
print('The score of RadioMap_TypeI is %f bps/Hz' % data_rate)
