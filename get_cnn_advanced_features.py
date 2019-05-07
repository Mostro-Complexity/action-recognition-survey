import os

import cv2
import h5py
import numpy as np
from keras import backend
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, confusion_matrix
import hdf5storage


DATASET = ['MSRAction3D']

N_ACTIONS, N_SUBJECTS, N_INSTANCES = 20, 10, 3
VIDEO_LENGTH, IM_LENGTH, IM_WIDTH = 38, 32, 32


def load_dataset_param(splits_path, skeletal_data_path):
    """Train-test validation info and the validity of dataset

    Arguments:
        splits_path {str} -- File path of train-test validation info 
        skeletal_data_path {str} -- File path of validity info of dataset

    Returns:
        tr_subjects {numpy.array} -- Training subjects indices
        te_subjects {numpy.array} -- Test subjects indices
        validity {numpy.array} -- Validity of each example
    """
    f = h5py.File(splits_path, 'r')
    tr_subjects = f['tr_subjects'][:].T
    te_subjects = f['te_subjects'][:].T

    f = h5py.File(skeletal_data_path, 'r')
    validity = f['skeletal_data_validity'][:]

    return tr_subjects, te_subjects, validity


def save_eval_results(output_dir, total_accurary, confusion_matrics):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    avg_total_accurary = total_accuracy.mean(axis=-1)
    avg_confusion_matrix = confusion_matrics.mean(axis=0)

    matfiledata = {
        u'total_accurary': total_accurary,
        u'confusion_matrics': confusion_matrics,
        u'avg_total_accurary': avg_total_accurary,
        u'avg_confusion_matrix': avg_confusion_matrix
    }
    hdf5storage.savemat('/'.join([output_dir, 'classification_results.mat']),
                        matfiledata)


def get_videos_info(validity):
    n_valid = len(validity[validity == 1])
    video_names = np.empty(n_valid, dtype=object)
    action_labels = np.empty(n_valid, dtype=int)
    subject_labels = np.empty(n_valid, dtype=int)
    instance_labels = np.empty(n_valid, dtype=int)
    count = 0

    for a in range(N_ACTIONS):
        for s in range(N_SUBJECTS):
            for e in range(N_INSTANCES):
                if validity[e, s, a] == 1:
                    video_names[count] = "a%02d_s%02d_e%02d_sdepth.mat" % (
                        a + 1, s + 1, e + 1)
                    # indices(labels) start from 1
                    action_labels[count] = a + 1
                    subject_labels[count] = s + 1
                    instance_labels[count] = e + 1
                    count += 1

    return video_names, action_labels, subject_labels, instance_labels


def split_subjects(video_info, tr_subjects, te_subjects):
    video_names, action_labels, subject_labels, instance_labels = video_info

    tr_subject_ind = np.isin(subject_labels, tr_subjects)
    te_subject_ind = np.isin(subject_labels, te_subjects)

    tr_labels = action_labels[tr_subject_ind]
    te_labels = action_labels[te_subject_ind]

    tr_names = video_names[tr_subject_ind]
    te_names = video_names[te_subject_ind]
    return tr_names, tr_labels, te_names, te_labels


def batches_generator(video_dir, video_names, video_labels, n_classes, batch_size):
    n_sequence = len(video_names)

    X_batch = np.empty((batch_size, 1, VIDEO_LENGTH,
                        IM_LENGTH, IM_WIDTH), dtype=float)
    y_batch = np.empty((batch_size, n_classes), dtype=int)

    while True:
        for i in range(int(np.ceil(n_sequence / batch_size))):
            # print('genetating batch %d' % (i + 1))

            if (i + 1) * batch_size > n_sequence:  # last one batch
                for j in range(n_sequence - i * batch_size):
                    filepath = os.path.join(
                        video_dir, video_names[i * batch_size + j])
                    f = h5py.File(filepath, 'r')
                    video = f['video_array'][:].swapaxes(1, 2)

                    X_batch[j, 0, :, :, :] = video
                    y_batch[j, :] = video_labels[i * batch_size + j]

                yield X_batch[:n_sequence - i * batch_size], y_batch[:n_sequence - i * batch_size]
                continue  # end the loop once

            for j in range(batch_size):
                filepath = os.path.join(
                    video_dir, video_names[i * batch_size + j])
                f = h5py.File(filepath, 'r')
                video = f['video_array'][:].swapaxes(1, 2)

                X_batch[j, 0, :, :, :] = video
                y_batch[j, :] = video_labels[i * batch_size + j]

            yield X_batch, y_batch


def network_construction():
    model = Sequential()

    model.add(Convolution3D(
        32,  # number of kernel
        kernel_dim1=7,  # depth
        kernel_dim2=5,  # rows
        kernel_dim3=5,  # cols
        input_shape=(1, n_frames, img_row, img_col),
        activation='relu',
        strides=1,
        padding='same',
        name='CL1'
    ))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), name='MP1'))
    model.add(Convolution3D(
        128,  # number of kernel
        kernel_dim1=5,  # depth
        kernel_dim2=5,  # rows
        kernel_dim3=5,  # cols
        activation='relu',
        strides=1,
        padding='same',
        name='CL2'
    ))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), name='MP2'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2056, init='normal', activation='linear', name='FC1'))
    model.add(Dropout(0.5))
    model.add(Dense(512, init='normal', activation='linear', name='FC2'))
    model.add(Dropout(0.5))
    model.add(Dense(128, init='normal', activation='linear', name='FC3'))
    # model.add(Dropout(0.5))
    model.add(Dense(n_classes, init='normal'))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.0001),
                  metrics=['mse', 'accuracy'])

    feature_model = Model(inputs=model.input,
                          outputs=model.get_layer('FC2').output)
    return model, feature_model


if __name__ == "__main__":
    # Channels order
    backend.set_image_dim_ordering('th')

    for dataset in DATASET:
        output_dir = dataset + '_experiments/advanced_features'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        features_dir = ''.join(['data/', dataset, '/Advanced_Feat_Mat'])
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)

        video_dir = ''.join(['data/', dataset, '/Depth_Mat'])

        splits_path = ''.join(['data/', dataset, '/tr_te_splits.mat'])
        skeletal_data_path = ''.join(['data/', dataset, '/skeletal_data.mat'])

        tr_subjects, te_subjects, validity = load_dataset_param(
            splits_path, skeletal_data_path)

        video_names, action_labels, subject_labels, instance_labels = get_videos_info(
            validity)

        video_info = (video_names, action_labels,
                      subject_labels, instance_labels)

        n_tr_te_splits = tr_subjects.shape[0]

        batch_size = 80
        img_row, img_col, n_frames = 32, 32, 38
        unique_classes = np.unique(action_labels)
        n_classes = len(np.unique(action_labels))

        confusion_matrics = np.empty(
            (n_tr_te_splits, n_classes, n_classes), dtype=float)
        total_accuracy = np.empty(n_tr_te_splits, dtype=float)

        for i in range(n_tr_te_splits):
            tr_names, tr_labels, te_names, te_labels = split_subjects(
                video_info, tr_subjects[i, :], te_subjects[i, :])

            # convert class vectors to binary class matrices
            tr_labels = np_utils.to_categorical(tr_labels - 1, n_classes)
            te_labels = np_utils.to_categorical(te_labels - 1, n_classes)

            n_steps = np.ceil(tr_labels.shape[0] / batch_size)

            # network training
            gen = batches_generator(
                video_dir, tr_names, tr_labels, n_classes, batch_size)

            model, feature_model = network_construction()
            model.fit_generator(generator=gen, epochs=60,
                                steps_per_epoch=n_steps)

            # features extraction
            n_steps = np.ceil(action_labels.shape[0] / batch_size)
            gen = batches_generator(
                video_dir, video_names, action_labels, n_classes, batch_size)
            advanced_features = feature_model.predict_generator(
                generator=gen, steps=n_steps)
            hdf5storage.savemat(
                features_dir + '/advanced_split_%d' % (i + 1),
                {u'advanced_features': advanced_features}
            )

            # network evaluation
            n_steps = np.ceil(te_labels.shape[0] / batch_size)

            gen = batches_generator(
                video_dir, te_names, te_labels, n_classes, batch_size)

            metrics_values = model.evaluate_generator(gen, steps=n_steps)
            print('Metrics Names:\n', model.metrics_names,
                  '\nMetrics Values:\n', metrics_values)

            # network prediction
            gen = batches_generator(
                video_dir, te_names, te_labels, n_classes, batch_size)
            pr_labels = model.predict_generator(generator=gen, steps=n_steps)

            # convert binary class matrices to class vectors
            te_labels = np.argmax(te_labels, axis=-1) + 1
            pr_labels = np.argmax(pr_labels, axis=-1) + 1

            total_acc = accuracy_score(te_labels, pr_labels)
            print('Split %d finished,total accuracy:%f' % (i + 1, total_acc))
            total_accuracy[i] = total_acc

            cm = confusion_matrix(te_labels, pr_labels, labels=unique_classes)
            cm = cm.astype(float) / cm.sum(axis=-1)[:, np.newaxis]
            confusion_matrics[i, :, :] = cm

            # avoiding OOM
            backend.clear_session()

        print('Average accuracy:%f' % total_accuracy.mean(axis=-1))
        save_eval_results(output_dir + '/conv_net',
                          total_accuracy, confusion_matrics)
    pass
