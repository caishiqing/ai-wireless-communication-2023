# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import scipy.io as scio

# from modelDesign import *


# # Parameters
# SC_num = 128  # subcarrier number
# Tx_num = 32  # Tx antenna number
# Rx_num = 4  # Rx antenna number
# sigma2_UE = 0.1

# # Read Data
# f = scio.loadmat('./data/train.mat')
# data = f['train']
# location_data = data[0, 0]['loc'][:4000]
# channel_data = data[0, 0]['CSI'][:4000]

# ##
# EPOCHS = 100
# LR = 1e-4
# opt_adam = keras.optimizers.Adam(learning_rate=LR)  # norm of the gradient coefficients

# ########################################################   Scheme 1   ########################################################
# # ANN for CSI estimation
# Input = keras.Input(shape=(3))
# Output = ANN_TypeI(Input)
# ANN_TypeI_model = keras.Model(inputs=Input, outputs=Output, name='ANN_TypeI')
# ANN_TypeI_model.compile(optimizer=opt_adam, loss='mse')
# ANN_TypeI_model.fit(x=location_data,
#                     y=channel_data,
#                     batch_size=128,
#                     epochs=EPOCHS,
#                     shuffle=True,
#                     verbose=2,
#                     validation_split=0.1)

# # Prediction
# channel_est = ANN_TypeI_model.predict(location_data, batch_size=32)
# PrecodingVector1 = DownPrecoding(channel_est)
# SubCH_gain_codeword_1 = EqChannelGain(channel_data, PrecodingVector1)
# data_rate1 = DataRate(SubCH_gain_codeword_1, sigma2_UE)

# # Generate Radio Map (Input:location, Output:beamforming vector)
# RadioMap_TypeI = RadioMap_Model_TypeI(ANN_TypeI_model, input=tf.keras.Input(shape=[3]))
# RadioMap_TypeI.compile(opt_adam, 'mse')
# RadioMap_TypeI.save('./RadioMap_TypeI.h5')
# ####
# ########################################################   Scheme 1   ########################################################


# ########################################################   Scheme 2   ########################################################
# # Define a beamforming codebook
# codebook_size = Tx_num
# DFTcodebook = np.zeros([codebook_size, codebook_size], dtype=complex)  # In this example, we define a DFT codebook
# for isub in range(codebook_size):
#     DFTcodebook[isub, :] = np.exp(isub * 1j * np.pi * 2 * np.arange(0, codebook_size, 1) / codebook_size)

# # Codeword-performance table
# SubCH_gain = np.zeros([channel_data.shape[0], SC_num, codebook_size])
# for ixx in range(codebook_size):
#     codeword_temp = DFTcodebook[ixx, :].reshape(-1, 1)
#     codeword_temp = np.expand_dims(codeword_temp, axis=0)
#     codeword_temp = np.repeat(codeword_temp, SC_num, axis=0)
#     codeword_temp = np.expand_dims(codeword_temp, axis=0)
#     codeword_temp = np.repeat(codeword_temp, channel_data.shape[0], axis=0)
#     SubCH_gain_codeword = EqChannelGain(channel_data, codeword_temp)
#     SubCH_gain[:, :, ixx] = SubCH_gain_codeword

# # ANN for codeword-performance table
# Input = keras.Input(shape=(3))
# Output = ANN_TypeII(Input)
# ANN_TypeII_model = keras.Model(inputs=Input, outputs=Output, name='ANN_TypeII')
# ANN_TypeII_model.compile(optimizer=opt_adam, loss='mse')
# ANN_TypeII_model.fit(x=location_data,
#                      y=SubCH_gain,
#                      batch_size=128,
#                      epochs=EPOCHS,
#                      shuffle=True,
#                      verbose=2,
#                      validation_split=0.1)

# # Prediction
# SubCH_gain_est = ANN_TypeII_model.predict(location_data)
# Beam_index = np.argmax(SubCH_gain_est, axis=-1)
# PrecodingVector2 = DFTcodebook[Beam_index, :]
# PrecodingVector2 = np.reshape(PrecodingVector2, [-1, SC_num, Tx_num, 1])
# SubCH_gain_codeword_2 = EqChannelGain(channel_data, PrecodingVector2)
# data_rate2 = DataRate(SubCH_gain_codeword_2, sigma2_UE)

# # Generate Radio Map (Input:location, Output:beamforming vector)
# RadioMap_TypeII = RadioMap_Model_TypeII(ANN_TypeII_model,
#                                         DFTcodebook,
#                                         input=tf.keras.Input(shape=[3]))
# RadioMap_TypeII.compile(opt_adam, 'mse')
# RadioMap_TypeII.save('./RadioMap_TypeII.h5')
# ####
# ########################################################   Scheme 2   ########################################################

# print('The score of RadioMapI is %.8f bps/Hz' % data_rate1)
# print('The score of RadioMapII is %.8f bps/Hz' % data_rate2)


from ModelDesign import preEncoding, trainModel
import tensorflow as tf
import scipy.io as scio
import argparse


f = scio.loadmat('./data/train.mat')
data = f['train']
location_data = data[0, 0]['loc']
channel_data = data[0, 0]['CSI']


parser = argparse.ArgumentParser()
parser.add_argument("--num_rx", type=int, default=4)
parser.add_argument("--num_tx", type=int, default=32)
parser.add_argument("--num_sc", type=int, default=128)
parser.add_argument("--sigma", type=float, default=0.1)
parser.add_argument("--hidden_dims", type=int, default=None)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--save_path", type=str, default="model.h5")
args = parser.parse_args()


model = preEncoding(num_tx=args.num_tx,
                    num_sc=args.num_sc,
                    num_layers=args.num_layers,
                    hidden_dims=args.hidden_dims)

csi_shape = (args.num_rx, args.num_tx, args.num_sc, 2)
train_model = trainModel(model, csi_shape, args.sigma)
train_model.compile(tf.keras.optimizers.Adam(args.learning_rate))
train_model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(args.save_path,
                                                monitor="val_snr",
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode="max")

train_model.fit(x=(location_data, channel_data),
                batch_size=args.batch_size,
                epochs=args.epochs,
                callbacks=[checkpoint],
                validation_split=0.1,
                verbose=2)

train_model.load_weights(args.save_path)
model.save(args.save_path)
