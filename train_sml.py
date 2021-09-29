import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

from loss import permutation_invariant_loss
from model import TasnetWithDprnn
from callbacks import EpochCheckpoint_Two_Models, ModelCheckpoint_and_Reduce_LR

def parse_args():
    parser = argparse.ArgumentParser(description='Training deep mutual learning for speech separation')
    parser.add_argument('--num_speaker', type=int, default=2, help='Do not change')
    parser.add_argument('--len_in_seconds', type=int, default=4, help='Length in seconds')
    parser.add_argument('--sample_rate_hz', type=int, default=8000, help='sample rate in hz')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate init')
    parser.add_argument('--alpha', type=float, default=0.001, help='alpha factor')
    parser.add_argument('--confidence', type=float, default=-15.0, help='confidence factor')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_path', type=str, default='save_model', help='save_model')
    args = parser.parse_args()
    return args

class TasNet_DML(Model):
    def __init__(self, model1, model2, alpha=0.1, confidence=-13.):
        super(TasNet_DML, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.alpha = alpha
        self.confidence = confidence

    def compile(self, optimizer, loss):
        super(TasNet_DML, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    @tf.function
    def train_step(self, data):
        x, y = data

        # Using model 2 as a pseudo-teacher
        y_model2 = self.model2(x, training=False)
        y_model2 = tf.stop_gradient(y_model2)
        temp_loss = tf.stop_gradient(self.loss(y, y_model2))
        pseudo_mask = tf.cast(temp_loss <= self.confidence, dtype=tf.float32)

        #--------Start training cho model 1-------------
        with tf.GradientTape() as m1_tape:
            # Forward pass of model 1
            y_model1 = self.model1(x, training=True)

            # Compute loss
            loss_sisnr_m1 = self.loss(y, y_model1)
            loss_dml_m1 = self.loss(y_model2, y_model1)
            loss_dml_m1 = pseudo_mask * loss_dml_m1
            loss_m1 = loss_sisnr_m1 + self.alpha * loss_dml_m1
            loss_m1 += sum(self.model1.losses)

        # Compute gradients for model 1
        trainable_vars_m1 = self.model1.trainable_variables
        gradients_m1 = m1_tape.gradient(loss_m1, trainable_vars_m1)
        # Update weights for student 1
        self.optimizer.apply_gradients(zip(gradients_m1, trainable_vars_m1))

        #-------------------------------------------------------------------------------------------
        # Use the model 1 as the pseudo-teacher to guide the model 2
        y_model1 = tf.stop_gradient(y_model1)
        pseudo_mask = tf.cast(loss_sisnr_m1 <= self.confidence, dtype=tf.float32)

        # --------Start training for the model 2-------------
        with tf.GradientTape() as m2_tape:
            # Forward pass of model 2
            y_model2 = self.model2(x, training=True)
            # Compute loss
            loss_sisnr_m2 = self.loss(y, y_model2)
            loss_dml_m2 = self.loss(y_model1, y_model2)
            loss_dml_m2 = loss_dml_m2 * pseudo_mask
            loss_m2 = loss_sisnr_m2 + self.alpha * loss_dml_m2
            loss_m2 += sum(self.model2.losses)

        # Compute gradients for model 1
        trainable_vars_m2 = self.model2.trainable_variables
        gradients_m2 = m2_tape.gradient(loss_m2, trainable_vars_m2)
        # Update weights for student 1
        self.optimizer.apply_gradients(zip(gradients_m2, trainable_vars_m2))

        # Return a dict of performance
        results = {}
        results.update({"loss_sisnr_M1": loss_sisnr_m1, "loss_dml_M1": loss_dml_m1, "loss_M1": loss_m1,
                        "loss_sisnr_M2": loss_sisnr_m2, "loss_dml_M2": loss_dml_m2, "loss_M2": loss_m2})
        return results

    @tf.function
    def test_step(self, data):
        x, y = data

        y_m1 = self.model1(x, training=False)
        y_m2 = self.model2(x, training=False)

        # Calculate the loss for model 1
        m1_loss = self.loss(y, y_m1)
        # m1_loss += sum(self.model1.losses)
        # Calculate the loss for model 2
        m2_loss = self.loss(y, y_m2)
        # m2_loss += sum(self.model2.losses)

        s1_0 = y_m1[:, 0, :]
        s1_1 = y_m1[:, 1, :]
        s2_0 = y_m2[:, 0, :]
        s2_1 = y_m2[:, 1, :]

        # Case 1
        case1_0 = (s1_0 + s2_0) / 2.
        case1_1 = (s1_1 + s2_1) / 2.
        case1 = tf.stack([case1_0, case1_1], axis=1)
        ave_loss_1 = self.loss(y, case1)

        # Case 2
        case2_0 = (s1_0 + s2_1) / 2.
        case2_1 = (s1_1 + s2_0) / 2.
        case2 = tf.stack([case2_0, case2_1], axis=1)
        ave_loss_2 = self.loss(y, case2)

        ave_loss = tf.math.minimum(ave_loss_1, ave_loss_2)

        # Return a dict of performance
        results = {}
        results.update({"loss_M1": m1_loss, "loss_M2": m2_loss, "average": ave_loss})
        return results

def build_callbacks(save_path, patience=20, every=5, startAt=1, lr_init=0.001):
    checkpoint_path = os.path.join(save_path, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    epoch_checkpoint = EpochCheckpoint_Two_Models(outputPath=checkpoint_path, every=every, startAt=startAt)
    jsonPath = os.path.join(save_path, 'output')

    if not os.path.exists(jsonPath):
        os.mkdir(jsonPath)

    jsonName = 'log.json'
    Checkpoint_LR = ModelCheckpoint_and_Reduce_LR(folderpath=save_path, jsonPath=jsonPath, jsonName=jsonName,
                                                  startAt=startAt, patience=patience, LR_decay_every_N_epochs=2,
                                                  lr_init=lr_init, monitor='val_loss_M1', mode='min', LR_decay_factor=0.98,
                                                  n_epoch_add_confidence=15, current_confidence=-15,
                                                  verbose=1)
    cb = [epoch_checkpoint, Checkpoint_LR]
    return cb

def read_dataset(npz_file):
    d = np.load(npz_file)
    return d["x"], d["y"]

def train_network(num_speaker=2, samplerate_hz=8000, length_in_seconds=4, batch_size=32, lr_init=0.001, alpha=0.1, confidence=-13.0,
                  save_path='save_model', epochs=200, start_epoch=1):
    # Prepare data for training phase
    X_train, y_train = read_dataset('train_data.npz')
    X_val, y_val = read_dataset('val_data.npz')

    print('X train shape: ', X_train.shape)
    print('X_val shape: ', X_val.shape)

    # NETWORK PARAMETERS
    NETWORK_NUM_FILTERS_IN_ENCODER = 64
    NETWORK_ENCODER_FILTER_LENGTH = 2
    NETWORK_NUM_UNITS_PER_LSTM = 200
    NETWORK_NUM_DPRNN_BLOCKS = 3
    NETWORK_CHUNK_SIZE = 256
    train_num_full_chunks = samplerate_hz * length_in_seconds // NETWORK_CHUNK_SIZE

    # Build model 1
    tasnet_1 = TasnetWithDprnn(batch_size=batch_size,
                             model_weights_file=None,
                             num_filters_in_encoder=NETWORK_NUM_FILTERS_IN_ENCODER,
                             encoder_filter_length=NETWORK_ENCODER_FILTER_LENGTH,
                             chunk_size=NETWORK_CHUNK_SIZE,
                             num_full_chunks=train_num_full_chunks,
                             units_per_lstm=NETWORK_NUM_UNITS_PER_LSTM,
                             num_dprnn_blocks=NETWORK_NUM_DPRNN_BLOCKS,
                             samplerate_hz=samplerate_hz)

    # Build model 2
    tasnet_2 = TasnetWithDprnn(batch_size=batch_size,
                               model_weights_file=None,
                               num_filters_in_encoder=NETWORK_NUM_FILTERS_IN_ENCODER,
                               encoder_filter_length=NETWORK_ENCODER_FILTER_LENGTH,
                               chunk_size=NETWORK_CHUNK_SIZE,
                               num_full_chunks=train_num_full_chunks,
                               units_per_lstm=NETWORK_NUM_UNITS_PER_LSTM,
                               num_dprnn_blocks=NETWORK_NUM_DPRNN_BLOCKS,
                               samplerate_hz=samplerate_hz)

    # Load weights and continuous training
    if start_epoch > 1:
        print('--------------------------START LOAD WEIGHT FROM CURRENT EPOCH---------------------------------')
        path_1 = os.path.join(save_path, 'checkpoints', 'tasnet1_epoch_' + str(start_epoch) + '.h5')
        tasnet_1.model.load_weights(path_1)
        path_2 = os.path.join(save_path, 'checkpoints', 'tasnet2_epoch_' + str(start_epoch) + '.h5')
        tasnet_2.model.load_weights(path_2)
        print('--------------------------LOAD WEIGHT COMPLETED---------------------------------')

    # Build callbacks
    calbacks = build_callbacks(save_path, patience=15, every=1, startAt=start_epoch, lr_init=lr_init)

    # Start training the model
    optimizer_clip_l2_norm_value = 5
    adam = keras.optimizers.Adam(learning_rate=lr_init, clipnorm=optimizer_clip_l2_norm_value)
    step_train = len(X_train) // batch_size
    step_val = len(X_val) // batch_size

    # Training model
    TasNet_DML_model = TasNet_DML(model1=tasnet_1.model, model2=tasnet_2.model, alpha=alpha, confidence=confidence)
    TasNet_DML_model.compile(optimizer=adam, loss=permutation_invariant_loss)
    history = TasNet_DML_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val),
                         steps_per_epoch=step_train, validation_steps=step_val, callbacks=calbacks,
                                   initial_epoch=start_epoch if start_epoch>1 else 0)

    return history

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Choose GPU for training
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    num_speaker = args.num_speaker
    len_in_seconds = args.len_in_seconds
    sample_rate_hz= args.sample_rate_hz
    lr_init = args.lr
    start_epoch = args.start_epoch
    epochs = args.epochs
    batch_size = args.batch_size
    alpha = args.alpha
    confidence = args.confidence
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    train_network(num_speaker=num_speaker, samplerate_hz=sample_rate_hz, length_in_seconds=len_in_seconds, batch_size=batch_size, alpha=alpha,
                  confidence=confidence, lr_init=lr_init, save_path=save_path, epochs=epochs, start_epoch=start_epoch)

if __name__ == '__main__':
    print(tf.__version__)
    main()


