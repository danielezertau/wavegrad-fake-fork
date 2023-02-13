import soundfile
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import soundfile as sf
from scipy import linalg
import librosa


def _stable_trace_sqrt_product(sigma_test, sigma_train, eps=1e-7):
    sqrt_product, _ = linalg.sqrtm(sigma_test.dot(sigma_train), disp=False)
    if not np.isfinite(sqrt_product).all():
        offset = np.eye(sigma_test.shape[0]) * eps
        sqrt_product = linalg.sqrtm((sigma_test + offset).dot(sigma_train + offset))
    if not np.allclose(np.diagonal(sqrt_product).imag, 0, atol=1e-3):
        raise ValueError('sqrt_product contains large complex numbers.')
    sqrt_product = sqrt_product.real
    return np.trace(sqrt_product)


def get_embeddings(files, model, factor=False, rms=False, prefix=False):
    embedding_lst = []
    for f in files:
        print('file', f)
        # waveform = np.zeros(3 * 16000, dtype=np.float32)
        try:
            audio, sr = sf.read(f, dtype='int16')
        except:
            audio, sr = librosa.load(f, sr=16000)
            audio = (32768. * audio).astype(np.int16)
        if len(audio.shape) == 2:
            audio = audio.astype(float).mean(axis=1)
        else:
            audio = audio.astype(float)
        if prefix:
            audio = audio[: 16000 * prefix]
        waveform = audio
        waveform = waveform / 32768.

        if factor:
            waveform = np.clip(waveform * factor, -1, 1)

        # Run the model, check the output.
        embedding = model(waveform)
        embedding.shape.assert_is_compatible_with([None, 128])
        embedding_lst.append(embedding.numpy())
    return np.concatenate(embedding_lst, axis=0)


def frechet(mu1, sig1, mu2, sig2):
    diff = mu1 - mu2
    frob = np.dot(diff, diff)
    tr_sqrt = _stable_trace_sqrt_product(sig1, sig2)
    return frob + np.trace(sig1) + np.trace(sig2) - 2 * tr_sqrt


def soundstream(input_file_name, output_file_name):
    module = hub.KerasLayer('https://tfhub.dev/google/soundstream/mel/decoder/music/1')

    SAMPLE_RATE = 16000
    N_FFT = 1024
    HOP_LENGTH = 320
    WIN_LENGTH = 640
    N_MEL_CHANNELS = 128
    MEL_FMIN = 0.0
    MEL_FMAX = int(SAMPLE_RATE // 2)
    CLIP_VALUE_MIN = 1e-5
    CLIP_VALUE_MAX = 1e8

    MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MEL_CHANNELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=MEL_FMIN,
        upper_edge_hertz=MEL_FMAX)

    def calculate_spectrogram(samples):
        """Calculate mel spectrogram using the parameters the model expects."""
        fft = tf.signal.stft(
            samples,
            frame_length=WIN_LENGTH,
            frame_step=HOP_LENGTH,
            fft_length=N_FFT,
            window_fn=tf.signal.hann_window,
            pad_end=True)
        fft_modulus = tf.abs(fft)

        output = tf.matmul(fft_modulus, MEL_BASIS)

        output = tf.clip_by_value(
            output,
            clip_value_min=CLIP_VALUE_MIN,
            clip_value_max=CLIP_VALUE_MAX)
        output = tf.math.log(output)
        return output

    # Load a music sample from the GTZAN dataset.
    samples, _ = soundfile.read(input_file_name, dtype=np.float32)
    # Convert an example from int to float.
    # samples = tf.cast(audio / 32768, dtype=tf.float32)
    # Add batch dimension.
    samples = tf.expand_dims(samples, axis=0)
    # Compute a mel-spectrogram.
    spectrogram = calculate_spectrogram(samples)
    # Reconstruct the audio from a mel-spectrogram using a SoundStream decoder.
    reconstructed_samples = module(spectrogram).numpy().squeeze()
    sf.write(output_file_name, reconstructed_samples, samplerate=SAMPLE_RATE, format='flac', subtype="PCM_24")


if __name__ == '__main__':
    # Load the model.
    model = hub.load('https://tfhub.dev/google/vggish/1')
    print('loaded model')

    # Create SoundStream output file
    soundstream_output_file = './soundstream-cello.flac'
    soundstream_input_file = "/Users/danielezer/IdeaProjects/wavegrad-fake-fork/music-inf/Bach - Cello Suite No.5 6-Gigue Short.flac"
    soundstream(soundstream_input_file, soundstream_output_file)

    eval_files = [soundstream_output_file]  # model prediction

    bg_files = [soundstream_input_file]  # ground truth or target

    eval_embeddings = get_embeddings(eval_files, model)
    bg_embeddings = get_embeddings(bg_files, model)

    mu_e = eval_embeddings.mean(axis=0)
    sigma_e = np.cov(eval_embeddings, rowvar=False)

    mu_bg = bg_embeddings.mean(axis=0)
    sigma_bg = np.cov(bg_embeddings, rowvar=False)
    frechet_dist = frechet(mu_e, sigma_e, mu_bg, sigma_bg)
    print('frechet', frechet_dist)
    print('done.')
