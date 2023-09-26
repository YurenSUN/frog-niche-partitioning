import numpy as np
import scipy
import scipy.signal
import soundfile as sf
import h5py
import cv2

# Split label into smaller chunks to fit into the used time range of sample boxes
def split_time_interval(start_time, end_time, delta_time, sec_used):
    time_intervals = []
    if delta_time <= sec_used:
        time_intervals.append([start_time, end_time])
    else:
        # slice to chunks with max sec_used with average length
        seperated_num = delta_time // sec_used
        seperated_num += 1 if delta_time % sec_used else 0
        seperated_delta = delta_time / float(seperated_num)
        for i in range(int(seperated_num)):
            cur_time = start_time + i * seperated_delta
            time_intervals.append([cur_time, min(end_time, cur_time + seperated_delta)])

    return time_intervals


def retrieve_spectrogram(audio_path, max_freq=-1, one_sided=False):
    try:
        audio, rate = sf.read(audio_path)
        if one_sided:
            audio = audio[:, 0]
    except:
        print("ERROR READING FILE: %s" % audio_path)
        return [], [], []

    freq, t, spec = scipy.signal.spectrogram(audio, rate, axis=0, window="hann",
                                             nperseg=512, nfft=512, 
                                             noverlap=256)
    if max_freq < 0:
        # max_freq not provided
        return freq, t, spec

    if freq[-1] < max_freq:
        # The provided frequency is less than intended, not use so far, should
        # not have any file of this type, print alert in this case.
        print("File with frequency less than intended:", audio_path, freq[-1])
        return [], [], []
    else:
        freq = freq[: np.argmin(np.abs(freq - max_freq)) + 1]
    return freq, t, spec


def find_time_index(start, end, t):
    start_ind = np.argmin(np.abs(t - start))
    end_ind = np.argmin(np.abs(t - end))
    # adjust the params to cover the whole box and be aware of index out of bound
    if t[start_ind] > start:
        start_ind = max(start_ind - 1, 0)
    if t[end_ind] < end:
        end_ind = min(end_ind + 1, len(t) - 1)
    return start_ind, end_ind


def resize_spec(spec, x=224, y=224, x_ratio=1, y_ratio=1, log_scale = False):
    if log_scale:
        spec  = np.where(spec > 1.0e-10, spec, 1.0e-10)
        spec = 10*np.log10(spec)
    spec_resized = cv2.resize(
        spec.astype("float32"), (int(x * x_ratio), int(y * y_ratio))
    )
    # Pad into expected dim
    x_before = int((x - x * x_ratio) // 2)
    x_after = x - x_before - int(x * x_ratio)
    y_before = int((y - y * y_ratio) // 2)
    y_after = y - y_before - int(y * y_ratio)
    # Pad the min of spec if using logscale (goes to negative).
    # min_spec = np.min(spec) if log_scale else 0
    min_spec = np.min(spec)
    spec_resized = np.pad(
        spec_resized, ([0, 0], [x_before, x_after]), mode="constant", 
        constant_values=min_spec
    )
    # Resize into dim of 3
    spec_resized = np.expand_dims(spec_resized, 2)
    spec_resized = cv2.cvtColor(
        spec_resized.astype("float32"), cv2.COLOR_BGR2RGB
    )  # cv2 does not accept float64
    spec_resized = np.flip(spec_resized, 0)
    return spec_resized


def add_to_hdf5(hdf5_file, spec_resized, frog_audio_folder, sample_source_to_add):
    # add to hdf5
    f = h5py.File(hdf5_file, "a")
    f["specs"].resize((f["specs"].shape[0] + 1), axis=0)
    f["specs"][-1:, :, :, :] = [spec_resized]

    f["labels"].resize((f["labels"].shape[0] + 1), axis=0)
    f["labels"][-1:] = np.array([frog_audio_folder], dtype="S")

    f["sample_source"].resize((f["sample_source"].shape[0] + 1), axis=0)
    f["sample_source"][-1:] = np.array([sample_source_to_add], dtype="S")
    f.close()
