import numpy as np
import os
import sklearn


def get_cls_dict(cls):
    cls_list = sorted(np.unique(cls))
    cls_range = len(cls_list)
    cls_dict = {}
    for i in range(cls_range):
        cls_dict[cls_list[i]] = i
    return cls_dict


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def clean_cls(cls, cls_dict):
    """
    For some particular datasets, the labels are not continuous
    Like 0, 1, 3, 4, 5
    Using this function to clean the labels
    """
    cls_list = []
    for c in cls:
        cls_list.append(cls_dict[c])
    return np.array(cls_list, int)


def read_data(root_dir, data_name, normalization=True):
    """
    Function for reading UCR time series dataset
    the root_dir should be like '/path/to/UCR_TS_Archive_2015'
    """

    data_dir = os.path.join(root_dir, data_name)
    # print(data_dir)
    # print(os.path.join(data_dir, data_name + '_TRAIN.tsv'))
    # data_train = np.loadtxt(os.path.join(data_dir, data_name + '_TRAIN.tsv'))
    # data_test = np.loadtxt(os.path.join(data_dir, data_name + '_TEST.tsv'))
    data_train = np.loadtxt(os.path.join(data_dir, data_name + '_TRAIN.csv'), delimiter=',')
    data_test = np.loadtxt(os.path.join(data_dir, data_name + '_TEST.csv'), delimiter=',')
    data_train = np.delete(data_train, 0, axis=0)
    data_test = np.delete(data_test, 0, axis=0)

    cls_dict = get_cls_dict(data_train[:, 0])

    label_train = clean_cls(data_train[:, 0], cls_dict)
    label_test = clean_cls(data_test[:, 0], cls_dict)

    nb_classes = len(np.unique(np.concatenate((label_train, label_test), axis=0)))

    # save orignal y because later we will use binary
    label_true = label_test.astype(np.int64)
    label_true_train = label_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((label_train, label_test), axis=0).reshape(-1, 1))
    label_train = enc.transform(label_train.reshape(-1, 1)).toarray()
    label_test = enc.transform(label_test.reshape(-1, 1)).toarray()

    if normalization:
        mean_v = data_train[:, 1:].mean()
        std_v = data_train[:, 1:].std()

        input_train = np.array((data_train[:, 1:] - mean_v) / std_v, np.float32)
        input_test = np.array((data_test[:, 1:] - mean_v) / std_v, np.float32)
    else:
        input_train = np.array(data_train[:, 1:], np.float32)
        input_test = np.array(data_test[:, 1:], np.float32)

    return np.expand_dims(input_train, axis=2), label_train, np.expand_dims(input_test, axis=2), label_test, label_true, nb_classes, label_true_train


def padding_zeros(data_in, num_pad):
    print(f"padding zeros".center(90, '*'))
    print(f"Before: {data_in.shape}".center(50, '-'))
    a_shape = (num_pad,) + data_in.shape[1:]
    a = np.zeros(a_shape)
    print(a.shape)
    b = np.append(data_in, a, 0)
    print(f"End: {b.shape}".center(50, '-'))
    return b


def load_data(data_root, name_data, batch_size):
    x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train = read_data(data_root, name_data)

    x_train, y_train, y_true_train = batch_filled(batch_size, x_train, y_train, y_true_train)
    x_test, y_test, y_true = batch_filled(batch_size, x_test, y_test, y_true)

    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train


def batch_filled(batch_size, data, label, one_hot):
    num_padding = batch_size - (data.shape[0]) % batch_size
    if num_padding < batch_size:
        data = padding_zeros(data, num_padding)
        label = padding_zeros(label, num_padding)
        one_hot = padding_zeros(one_hot, num_padding)
        # print(f"data: {data.shape}".center(30, '-'))
        # print(f"label: {label.shape}".center(30, '-'))
        # print(f"one hot: {one_hot.shape}".center(30, '*'))
    return data, label, one_hot

