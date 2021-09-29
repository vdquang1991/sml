import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # Choose GPU for training

import numpy as np
from model import TasnetWithDprnn
from tensorflow import keras

from train_sml import read_dataset
from loss import get_permutation_invariant_sisnr
from pystoi import stoi
from pesq import pesq
from mir_eval import separation as sp

batch_size = 1
samplerate_hz = 8000
length_in_seconds = 4

num_sample_test = 3000

# NETWORK PARAMETERS
NETWORK_NUM_FILTERS_IN_ENCODER = 64
NETWORK_ENCODER_FILTER_LENGTH = 2
NETWORK_NUM_UNITS_PER_LSTM = 200
NETWORK_NUM_DPRNN_BLOCKS = 3
NETWORK_CHUNK_SIZE = 256
train_num_full_chunks = samplerate_hz * length_in_seconds // NETWORK_CHUNK_SIZE

tasnet_1 = TasnetWithDprnn(batch_size=batch_size,
                             model_weights_file=None,
                             num_filters_in_encoder=NETWORK_NUM_FILTERS_IN_ENCODER,
                             encoder_filter_length=NETWORK_ENCODER_FILTER_LENGTH,
                             chunk_size=NETWORK_CHUNK_SIZE,
                             num_full_chunks=train_num_full_chunks,
                             units_per_lstm=NETWORK_NUM_UNITS_PER_LSTM,
                             num_dprnn_blocks=NETWORK_NUM_DPRNN_BLOCKS,
                             samplerate_hz=samplerate_hz)

# Load weights
# save_path = 'alpha_0p1_epoch_191'
# path_1 = os.path.join(save_path, 'best_m1_model.h5')
path_1 = 'baseline.h5'
tasnet_1.model.load_weights(path_1)
print('Load model weights at ', path_1)
print('Load model weights completed!!!')

# Load X_test, y_test
X_val, y_val = read_dataset('val_data.npz')

X_val = X_val[:num_sample_test,:]
y_val = y_val[:num_sample_test,:,:]
print('Number of test samples: ', X_val.shape[0])

y_predict = tasnet_1.model.predict(X_val, batch_size=batch_size)

results = np.zeros((num_sample_test, 2))

stoi_result_list = []
pesq_result_list = []
sdri_list = []

for idx in range(num_sample_test):
    if idx%300==0:
        print('Processing %05d samples of %05d total' %(idx, num_sample_test))
    # X_sample = X_val[idx,:]
    # y_predict = tasnet_1.model.predict(X_sample)
    spk0_estimate = y_predict[idx, 0, :]
    spk1_estimate = y_predict[idx, 1, :]
    spk0_groundtruth = y_val[idx, 0, :]
    spk1_groundtruth = y_val[idx, 1, :]
    results[idx, 0], results[idx, 1] = get_permutation_invariant_sisnr(spk0_estimate, spk1_estimate, spk0_groundtruth, spk1_groundtruth)


    # Calculate STOI for speak 0
    stoi_1_a = stoi(spk0_groundtruth, spk0_estimate, 8000, extended=False)
    stoi_1_b = stoi(spk0_groundtruth, spk1_estimate, 8000, extended=False)
    if (stoi_1_a >= stoi_1_b):
        stoi_1 = stoi_1_a
    else:
        stoi_1 = stoi_1_b

    # Calculate STOI for speak 1
    stoi_2_a = stoi(spk1_groundtruth, spk0_estimate, 8000, extended=False)
    stoi_2_b = stoi(spk1_groundtruth, spk1_estimate, 8000, extended=False)
    if (stoi_2_a >= stoi_2_b):
        stoi_2 = stoi_2_a
    else:
        stoi_2 = stoi_2_b

    stoi_result = (stoi_1 + stoi_2) / 2.

    # Calculate PESQ
    pesq_1_a = pesq(8000, spk0_groundtruth, spk0_estimate, 'nb')
    pesq_1_b = pesq(8000, spk0_groundtruth, spk1_estimate, 'nb')
    if (pesq_1_a >= pesq_1_b):
        pesq_1 = pesq_1_a
    else:
        pesq_1 = pesq_1_b
    pesq_2_a = pesq(8000, spk1_groundtruth, spk0_estimate, 'nb')
    pesq_2_b = pesq(8000, spk1_groundtruth, spk1_estimate, 'nb')
    if (pesq_2_a >= pesq_2_b):
        pesq_2 = pesq_2_a
    else:
        pesq_2 = pesq_2_b
    pesq_result = (pesq_1 + pesq_2) / 2.

    # Calculate SDRI
    (sdr, sir, sar, _) = sp.bss_eval_sources(np.array([spk0_groundtruth, spk1_groundtruth]), np.array([spk0_estimate, spk1_estimate]))
    (sdr0, sir0, sar0, _) = sp.bss_eval_sources(np.array([spk0_groundtruth, spk1_groundtruth]),
                                                np.array([X_val[idx,:], X_val[idx,:]]))

    sdri = ((sdr[0] - sdr0[0]) + (sdr[1] - sdr0[1])) / 2

    stoi_result_list.append(stoi_result)
    pesq_result_list.append(pesq_result)
    sdri_list.append(sdri)

    # Print
    # print('Sample %05d has SISNR: %0.5f and %0.5f.' % (idx, results[idx, 0], results[idx, 1]))
    # print('Sample %05d has STOI: %0.5f' % (idx, stoi_result))
    # print('Sample %05d has PESQ: %0.5f' % (idx, pesq_result))
    # print('Sample %05d has SDRi: %0.5f' % (idx, sdri))
    # print('\n')


file_path = 'results.txt'
np.savetxt(file_path, results, fmt="%2.1f")

mean_results = np.mean(results)
mean_stoi = np.mean(np.asarray(stoi_result_list))
mean_pesq_result = np.mean(np.asarray(pesq_result_list))
mean_sdri = np.mean(np.asarray(sdri_list))

print('Mean results: ', mean_results)
print('Mean STOI: ', mean_stoi)
print('Mean PESQ: ', mean_pesq_result)
print('Mean SDRI: ', mean_sdri)


file_path = 'mean_results_baseline.txt'
with open(file_path, 'w') as f:
    s = 'Mean SI-SNR: ' + str(mean_results) + '\n'
    f.writelines(s)
    s = 'Mean STOI: ' + str(mean_stoi)  + '\n'
    f.writelines(s)
    s = 'Mean PESQ: ' + str(mean_pesq_result) + '\n'
    f.writelines(s)
    s = 'Mean SDRI: ' + str(mean_sdri) + '\n'
    f.writelines(s)
    f.close()
