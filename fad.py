import tensorflow_hub as hub
import numpy as np
import soundfile as sf
from scipy import linalg
import librosa
import os


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


def calc_fad(predictions, ground_truth):
    # Load VGGish.
    model = hub.load('https://tfhub.dev/google/vggish/1')

    eval_embeddings = get_embeddings([predictions], model)
    bg_embeddings = get_embeddings([ground_truth], model)

    mu_e = eval_embeddings.mean(axis=0)
    sigma_e = np.cov(eval_embeddings, rowvar=False)

    mu_bg = bg_embeddings.mean(axis=0)
    sigma_bg = np.cov(bg_embeddings, rowvar=False)
    frechet_dist = frechet(mu_e, sigma_e, mu_bg, sigma_bg)
    return frechet_dist


if __name__ == '__main__':
    model_results_directory = "./recon128"
    eval_files = [
        f"{model_results_directory}/output-The.wav",
        f"{model_results_directory}/output-Pink.wav",
        f"{model_results_directory}/output-David.wav",
        f"{model_results_directory}/output-Children's.wav",
        f"{model_results_directory}/output-Beethoven.wav",
        f"{model_results_directory}/output-Queen.wav",
        f"{model_results_directory}/output-Come.wav",
        f"{model_results_directory}/output-FranzSchubert-SonataInAMinorD.784-02-Andante.wav",
        f"{model_results_directory}/output-Here.wav",
        f"{model_results_directory}/output-Bach.wav"]  # model prediction

    inf_dir = "./music-inf"
    bg_files = [
        f"{inf_dir}/The Well Tempered Clavier, Book I, BWV 846-869 - Fugue No.2 in C minor, BWV 847 Short.flac",
        f"{inf_dir}/Pink Floyd - Money - Pink Floyd HD (Studio Version) Short.flac",
        f"{inf_dir}/David Bowie - Changes Short.flac",
        f"{inf_dir}/Children's Corner, L. 113 - I. Doctor Gradus ad Parnassum Short.flac",
        f"{inf_dir}/Beethoven - Symphony No. 9 in D minor, Op. 125 - II. Scherzo_ Molto Vivace - Presto Short.flac",
        f"{inf_dir}/Queen - I Want To Break Free Short.flac",
        f"{inf_dir}/Come Together (Remastered 2009) Short.flac",
        f"{inf_dir}/FranzSchubert-SonataInAMinorD.784-02-Andante Short.flac",
        f"{inf_dir}/Here Majesty (Remastered 2009).flac",
        f"{inf_dir}/Bach - Cello Suite No.5 6-Gigue Short.flac"]  # ground truth or target

    fad_results = []
    for predicted, reference in zip(eval_files, bg_files):
        fad_results.append(calc_fad(predicted, reference))
    np.set_printoptions(precision=3)
    fad_results = np.array(fad_results)
    print(f'{np.mean(fad_results)} +- {np.std(fad_results)}')
