import numpy as np
import os
import glob
import scipy.io.wavfile as wav
import scipy.special as special
import scipy.signal
import scipy
import itertools
import librosa.feature


def gcc_phat(spec, use_mel=False, nb_mel=96):
    """
    Computer GCC-phat across all channels
    Input:
        spec: input complex spectrogram, shape=[f,t,ch], f - number of frequency bins, t - number of time frames
        ch - number of channels
        use_mel (bool) - if true, function returns GCC-phat feature matrix cut to nb_mel in the frequency axis
        nb_mel - number of mel bands (used only if use_mel is True)
    """
    spec_shape = np.shape(spec)
    nb_ch = spec_shape[2]
    channels = np.arange(nb_ch)

    nb_gcc_ch = int(special.binom(nb_ch, 2))
    if not use_mel:
        gcc_phat = np.zeros((spec_shape[0], spec_shape[1], nb_gcc_ch))
    else:
        gcc_phat = np.zeros((nb_mel, spec_shape[1], nb_gcc_ch))

    idx = 0
    for l_ch, r_ch in itertools.combinations(channels, 2):
        R = spec[:, :, l_ch] * np.conj(spec[:, :, r_ch])
        cc = np.fft.irfft(np.exp(1.j * np.angle(R)), axis=0)

        if use_mel:
            cc = np.concatenate((cc[-nb_mel // 2:, :], cc[:nb_mel // 2, :]), axis=0)
        else:
            cc = np.concatenate((cc[-spec_shape[0] // 2:, :], cc[:spec_shape[0] // 2, :]), axis=0)

        gcc_phat[:, :, idx] = cc
        idx += 1

    return gcc_phat


def phase_diff(phase_spec, cos_sine=False):
    """
    Computes Interchannel Phase Differences for each channel pair
    Input:
        phase_spec: input phase spectrogram, shape=[f,t,ch], f - number of frequency bins, t - number of time frames
        ch - number of channels
        cos_sine (bool) - if true, function returns the sine&cosine variation of phase differences
    """
    spec_shape = np.shape(phase_spec)
    nb_ch = spec_shape[2]
    channels = np.arange(nb_ch)

    nb_diff_ch = int(special.binom(nb_ch, 2))

    if cos_sine:
        nb_diff_ch *= 2
        phase_diff = np.zeros((spec_shape[0], spec_shape[1], nb_diff_ch))
        idx = 0
        for l_ch, r_ch in itertools.combinations(channels, 2):
            diff = phase_spec[:, :, l_ch] - phase_spec[:, :, r_ch]
            phase_diff[:, :, idx] = np.cos(diff)
            phase_diff[:, :, idx + 1] = np.sin(diff)
            idx += 2

    else:
        phase_diff = np.zeros((spec_shape[0], spec_shape[1], nb_diff_ch))
        idx = 0
        for l_ch, r_ch in itertools.combinations(channels, 2):
            phase_diff[:, :, idx] = phase_spec[:, :, l_ch] - phase_spec[:, :,
                                                             r_ch]
            idx += 1

    return phase_diff


def ild(mag_spec):
    """
    Computes Interchannel Level Differences for each channel pair
    Input:
        mag_spec: input magnitude spectrogram, shape=[f,t,ch], f - number of frequency bins, t - number of time frames
        ch - number of channels
    """
    spec_shape = np.shape(mag_spec)
    nb_ch = spec_shape[2]
    channels = np.arange(nb_ch)

    nb_diff_ch = int(special.binom(nb_ch, 2))

    ild = np.zeros((mag_spec.shape[0], mag_spec.shape[1], nb_diff_ch))
    epsilon = 0.00000000001

    idx = 0
    for l_ch, r_ch in itertools.combinations(channels, 2):
        zero_mask = mag_spec[:, :, r_ch] == 0
        ild[:, :, idx] = np.where(zero_mask, 0, np.divide(mag_spec[:, :, l_ch], mag_spec[:, :, r_ch] + epsilon))
        idx += 1

    return ild


def mel_spectrogram(filepath, fs, n_fft, n_mels):
    # compute mel spec
    audio, sr = librosa.core.load(filepath, sr=fs, mono=False)
    mel_left = librosa.feature.melspectrogram(y=audio[0, :], sr=fs, n_fft=n_fft, n_mels=n_mels)
    mel_right = librosa.feature.melspectrogram(y=audio[1, :], sr=fs, n_fft=n_fft, n_mels=n_mels)
    mel = np.stack((mel_left, mel_right), axis=2)

    return mel

audiopath = './direction-dataset/audio/'
audiofiles = glob.glob('./direction-dataset/audio/*.wav')
# read audio file
for i in range(len(audiofiles)):

    filepath = audiofiles[i]
    save_fp = os.path.splitext(filepath)[0]
    fs, y = wav.read(filepath)
    ch_l = y[:, 0]
    ch_r = y[:, 1]

    nfft = 2048
    noverlap = nfft // 2
    # extract spectrogram for both channels
    signal_size = np.shape(y)
    nb_channels = signal_size[1]

    mag_specs = np.zeros(
        (nfft // 2 + 1, 2 * (signal_size[0] // nfft) - 1, nb_channels))
    phase_specs = np.zeros_like(mag_specs)
    complex_specs = np.zeros_like(mag_specs, dtype='complex')

    for i_ch in range(nb_channels):
        f, t, complex_specs[:, :, i_ch] = scipy.signal.spectrogram(y[:, i_ch],
                                                                   fs,
                                                                   window='hamming',
                                                                   nperseg=nfft,
                                                                   noverlap=noverlap,
                                                                   mode='complex')
        mag_specs[:, :, i_ch] = np.abs(
            complex_specs[:, :, i_ch])  # magnitude spectrogram
        phase_specs[:, :, i_ch] = np.angle(
            complex_specs[:, :, i_ch])  # phase spectrogram


    # compute some feature, e.g. GCC-phat
    gcc_spectrogram = gcc_phat(complex_specs)
    print(gcc_spectrogram.shape)
    # saving the spectrogram.
    np.save((save_fp + '_GCC'), gcc_spectrogram)

    # retrieving data from file.
    # gcc_load = np.load(save_fp + '_GCC.npy')

    # compute simple phase differences
    phase_diffs = phase_diff(phase_specs)
    print(phase_diffs.shape)
    np.save((save_fp + '_phase_diff'), phase_diffs)

    # compute sines & cosines
    phase_diffs_cossine = phase_diff(phase_specs, cos_sine=True)
    print(phase_diffs_cossine.shape)
    np.save((save_fp + '_phase_diffs_cossine'), phase_diffs_cossine)

    # compute interchannel level differences
    ilds = ild(mag_specs)
    print(ilds.shape)
    np.save((save_fp + '_ilds'), ilds)

    # Save mel spectrogram
    mel = mel_spectrogram(filepath, fs, n_fft=nfft, n_mels=128)
    print(f'Shape of mel {mel.shape}')
    np.save((save_fp + '_mel'), mel)

    # Save mag spec
    np.save((save_fp + '_magspec'), mag_specs)

    # Save mel-GCC
    mel_gcc = gcc_phat(complex_specs, use_mel=True, nb_mel=mel.shape[0])
    print(f' Shape of mel-GCC {mel_gcc.shape}')
    np.save((save_fp + '_mel_gcc_phat'), mel_gcc)


    cossine_gcc = np.concatenate((phase_diffs_cossine, gcc_spectrogram), axis=-1)
    print(f'Shape of cossine + gcc {cossine_gcc.shape}')
    np.save((save_fp + '_cossine_gcc'), cossine_gcc)


