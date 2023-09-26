import random
import os

import numpy as np
import pandas as pd
import cv2
from keras.utils import to_categorical
from pydub import AudioSegment  # mp3

from dataset_process_utils import *

# normalization.
def normalize(specs):
    return_specs = []
    for i in range(len(specs)):
        cur_spec = np.copy(specs[i])
        s_min = np.amin(cur_spec)
        s_max = np.amax(cur_spec)
        # specs[i] = (cur_spec - s_min)/(s_max - s_min) * 255
        return_specs.append((cur_spec - s_min) / (s_max - s_min))
        # return_specs.append( np.log( (cur_spec - s_min)/(s_max - s_min) ) )

    return return_specs


# Augmentations - Not used in final paper.
# choppint methods, each return a spec
# time chop
def time_chop(spec, rand_start, input_w=224, input_h=224):
    # chop_range = 0.1

    # chop out a random portion within the range
    # rand_start = random.randint(int(112* chop_range), int(244* chop_range))
    # while rand_start == 0:  # make sure not 0
    #   rand_start = random.randint(0, 224 * chop_range)

    time_chopped_spec = np.copy(spec)
    time_chopped_spec[:, input_h - rand_start :, :] = 0
    # print("time", time_chopped_spec.shape)

    return [time_chopped_spec]


# freq chop
def freq_chop(spec, rand_start, input_w=224, input_h=224):
    # frequency chop
    # chop_range = 0.1

    # chop out a random portion within the range
    # rand_start = random.randint(int(112* chop_range), int(244* chop_range))
    # while rand_start == 0:  # make sure not 0
    #   rand_start = random.randint(0, 224 * chop_range)

    freq_chopped_spec = np.copy(spec)
    freq_chopped_spec[0:rand_start, :, :] = 0
    # print("freq", freq_chopped_spec.shape)

    return [freq_chopped_spec]


# four side chop
def four_chop(spec, rand_start, input_w=224, input_h=224):
    # chopping from all four sides
    # chop_range = 0.1

    # chop out a random portion within the range
    # rand_start = random.randint(int(122* chop_range), int(244* chop_range))
    # while rand_start == 0:  # make sure not 0
    #   rand_start = random.randint(0, 244 * chop_range)
    four_chopped_spec = np.copy(spec)
    four_chopped_spec[0:rand_start, :, :] = 0  # top
    four_chopped_spec[:, input_h - rand_start :, :] = 0  # right
    four_chopped_spec[input_w - rand_start :, :, :] = 0  # bottom
    four_chopped_spec[:, 0:input_h, :] = 0  # left
    # print("four", four_chopped_spec.shape)

    return [four_chopped_spec]


# transalate up and down, return two spec
# frequency ranges too large[1333.532 2565.584] [3084.02  3890.613]
def translate(spec, roll_start, input_w=224, input_h=224):
    return_specs = []
    # use 10% as range
    # translate_range = 0.1

    # random portion within the range
    # roll_start = random.randint(1, int(244* translate_range))
    # while roll_start == 0:  # make sure not 0
    #   rand_start = random.randint(0, 224 * translate_range)

    return_specs.append(np.roll(spec, -roll_start, axis=0))
    return_specs.append(np.roll(spec, roll_start, axis=0))

    return return_specs


# widen and squeezing
def widen(spec, widen_index, input_w=224, input_h=224):
    return_specs = []
    # use 10% as range
    # widen_range = 0.1

    # random portion within the range
    # widen_index = random.randint(int(244* widen_range)/2, int(244* widen_range))

    widen_time_spec = cv2.resize(
        spec.astype("float32"), (input_h + widen_index, input_w)
    )
    widen_freq_spec = cv2.resize(
        spec.astype("float32"), (input_h, input_w + widen_index)
    )

    return_specs.append(
        widen_time_spec[:, widen_index // 2 : -widen_index // 2, :]
    )
    return_specs.append(
        widen_freq_spec[widen_index // 2 : -widen_index // 2, :, :]
    )

    return_specs = np.array(return_specs)
    # print("widen", return_specs.shape)

    return return_specs


def squeeze(spec, squeeze_index, input_w=224, input_h=224):
    # use 10% as range
    # squeeze_range = 0.1

    # random portion within the range
    # squeeze_index = random.randint(int(244* squeeze_range)/2, int(244* squeeze_range))

    squeezed = cv2.resize(
        spec.astype("float32"),
        (input_h - squeeze_index, input_w - squeeze_index),
    )
    squeeze_spec = np.zeros([input_w, input_h, 3])
    squeeze_spec[
        squeeze_index // 2 : -squeeze_index // 2,
        squeeze_index // 2 : -squeeze_index // 2,
        :,
    ] = squeezed

    # print("squeeze", squeeze_spec.shape)
    return [squeeze_spec]


# add noises methods, return a list of spec
def add_noises(spec, cur_label, labels_h5, typeUsed, specs_h5):
    # add noise from light rian -2, rain -3, heavy rain -4, thunder -5, aircraft -6, chainsaw -7, and car/truck -8
    return_specs = []
    # noise from all other labels
    allTypes = np.unique(labels_h5)
    noise_labels = [x for x in allTypes if x != typeUsed[cur_label]]

    for i in range(len(noise_labels)):
        noises_index = np.argwhere(labels_h5 == noise_labels[i]).flatten()

        noises = specs_h5[noises_index]
        index = random.randint(0, len(noises_index) - 1)
        noise = normalize(np.array(specs_h5[noises_index[index]]) / 3)

        return_specs.append(np.add(normalize([spec])[0], noise))

    return return_specs, len(noise_labels)


def augment(
    specs,
    labels,
    aug_num,
    typeUsed,
    specs_h5,
    augment_range=0.1,
    input_w=224,
    input_h=224,
):
    # augment_range = 0.1
    augment_specs_func = []
    augment_label_func = []

    for i in range(len(specs)):
        # generate random index array for augmentation
        indices = np.arange(
            int(input_w * augment_range / 3 * 2), int(input_h * augment_range)
        )
        np.random.shuffle(indices)
        indices = indices[:aug_num]

        # augment each spec and add to list
        cur_spec = np.copy(specs[i])
        # add itself to the list
        if len(augment_specs_func):
            augment_specs_func = np.append(
                augment_specs_func, [cur_spec], axis=0
            )
        else:
            augment_specs_func.append(cur_spec)
        # augment_specs_func.append(cur_spec)

        for index in indices:
            # print(index)
            # chop
            augment_specs_func = np.append(
                augment_specs_func,
                time_chop(
                    np.copy(cur_spec), index, input_w=input_w, input_h=input_h
                ),
                axis=0,
            )
            augment_specs_func = np.append(
                augment_specs_func,
                freq_chop(
                    np.copy(cur_spec), index, input_w=input_w, input_h=input_h
                ),
                axis=0,
            )
            augment_specs_func = np.append(
                augment_specs_func,
                four_chop(
                    np.copy(cur_spec), index, input_w=input_w, input_h=input_h
                ),
                axis=0,
            )

            # widen + squeeze
            augment_specs_func = np.append(
                augment_specs_func,
                squeeze(
                    np.copy(cur_spec), index, input_w=input_w, input_h=input_h
                ),
                axis=0,
            )
            augment_specs_func = np.append(
                augment_specs_func,
                widen(
                    np.copy(cur_spec), index, input_w=input_w, input_h=input_h
                ),
                axis=0,
            )

            # # noise
            noise_aug_num = 0
            # noise_spec, noise_aug_num= add_noises(np.copy(cur_spec), labels[i], typeUsed, specs_h5)
            # augment_specs_func = np.append(augment_specs_func, noise_spec, axis = 0)

            # translate
            augment_specs_func = np.append(
                augment_specs_func,
                translate(
                    np.copy(cur_spec), index, input_w=input_w, input_h=input_h
                ),
                axis=0,
            )

        augment_label_func = np.append(
            augment_label_func,
            np.repeat(labels[i], 1 + (8 + noise_aug_num) * aug_num),
            axis=0,
        )

    return augment_specs_func, augment_label_func


def gen(specs, labels, typeUsed, step_len=4, shape_x=224, shape_y=224):
    while 1:
        # shuffle data
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        # # use 30 of all data per epoch
        # indices = indices[:30]
        # step_len = 4

        for i in range(len(specs) // step_len):
            step_min = i * step_len
            step_max = min((i + 1) * step_len, len(specs))
            augment_specs, augment_sono = augment(
                specs[indices][step_min:step_max],
                labels[indices][step_min:step_max],
                1,
                input_w=shape_x,
                input_h=shape_y,
            )
            augment_specs_normal = normalize(augment_specs)
            cat_y_train = to_categorical(
                augment_sono, num_classes=len(typeUsed)
            )
            # print(len(np.array([augment_specs_normal])))
            # print(len(np.array([cat_y_train])))

            yield np.array([augment_specs_normal])[0], np.array([cat_y_train])[
                0
            ]


def upsample(
    specs,
    target_num,
    input_w=224,
    input_h=224,
    augment_range=0.1,
):
    num_to_upsample = target_num - len(specs)
    if num_to_upsample <= 0:
        return specs

    sample_funcs = np.array([time_chop, freq_chop, four_chop, squeeze, widen])
    # Randomly select functions to up sample
    selected_funcs = np.random.choice(sample_funcs, num_to_upsample)
    sample_indices = np.random.choice(np.arange(len(specs)), num_to_upsample)
    selected_aug_samples = specs[sample_indices]
    # if sample_sources is not None:
    #     selected_aug_sources = sample_sources[sample_indices]
    # Augmenataion index.
    indices = np.arange(
        int(input_w * augment_range / 3 * 2), int(input_h * augment_range)
    )
    indices = np.random.choice(indices, num_to_upsample)

    for i, (cur_func, cur_spec, index) in enumerate(
        zip(selected_funcs, selected_aug_samples, indices)
    ):
        augmented_samples = cur_func(
            np.copy(cur_spec), index, input_w=input_w, input_h=input_h
        )
        specs = np.append(
            specs, np.array([random.choice(augmented_samples)]), axis=0
        )
        # modified_source = cur_source + "augmented:" + cur_func.__name__
        # if sample_sources is not None:
        #     sample_sources = np.append(sample_sources, np.array([modified_source]), axis=0)

    return specs


# For classification pipeline, prediction from mp3 files.
def mp32wav(mp3_file, wav_name):
    # print("-Converting:", mp3_file)
    try:
        audio = AudioSegment.from_file(mp3_file)
    except:
        print("Error reading file {}, skipping".format(mp3_file))
        return False

    # set to mono
    audio = audio.set_channels(1)
    # set to 44.1 KHz
    audio = audio.set_frame_rate(44100)
    # save as wav
    if not os.path.exists(wav_name):
        audio.export(wav_name, format="wav")
        # print("-Wav file saved to:", wav_name)
    else:
        print("-Wav file exists:", wav_name)
    return True


def retrieve_spec_from_mp3(
    mp3_file, temp_folder, max_freq, sec_used, log_scale=False
):
    wav_name = temp_folder + mp3_file.split("/")[-1] + ".wav"
    success = mp32wav(mp3_file, wav_name)
    if not success:
        return np.array([]), []

    # Spectrogram in 1s intervals
    spec_to_predict = []
    freq, t, spec = retrieve_spectrogram(wav_name, max_freq=max_freq)
    cur_intervals = split_time_interval(0, t[-1], t[-1], sec_used)
    for [start, end] in cur_intervals:
        start_ind, end_ind = find_time_index(start, end, t)
        cur_spec = np.copy(spec)[: len(freq), start_ind:end_ind]
        spec_resized = resize_spec(
            cur_spec, x_ratio=(end - start) / sec_used, log_scale=log_scale
        )
        spec_to_predict.append(spec_resized)

    spec_to_predict = np.array(spec_to_predict)
    spec_to_predict = np.array(normalize(spec_to_predict))

    # Clean wav
    if os.path.exists(wav_name):
        os.remove(wav_name)

    return spec_to_predict, cur_intervals


def predict_mp3(
    mp3_file,
    temp_folder,
    model,
    typeUsed,
    max_freq,
    sec_used,
    spec_to_predict=None,
    cur_intervals=None,
    repeat=True,
):
    if spec_to_predict is None:
        spec_to_predict, cur_intervals = retrieve_spec_from_mp3(
            mp3_file, temp_folder, max_freq, sec_used
        )
    y_scores = model.predict(spec_to_predict, verbose=0)
    y_predict = np.argmax(y_scores, 1)
    type_predict = np.array([typeUsed[i] for i in y_predict])
    confidence = [y_scores[i][y_predict[i]] for i in range(len(y_predict))]
    if repeat:
        file_names_to_dump = np.repeat(mp3_file, len(y_scores))
    else:
        file_names_to_dump = mp3_file
    df = pd.DataFrame(
        [
            file_names_to_dump,
            cur_intervals,
            type_predict,
            confidence,
            y_scores,
            y_predict,
        ]
    ).T

    # Dump to pandas
    df = df.set_axis(
        [
            "file_path",
            "interval",
            "predicted_type",
            "confidience",
            "y_scores",
            "y_predict",
        ],
        axis=1,
        copy=False,
    )
    return df
