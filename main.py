import argparse
import os
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=10000, help='Decay step for lr decay [default: 50000]')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--data_name', type=str, default='Yoga', help='Name of UCR data [default: yoga]')
    parser.add_argument('--drop_rate_input', type=float, default=0.1,
                        help='Drop out rate for input layer[default: 0.1]')
    parser.add_argument('--drop_rate_hidden', type=float, default=0,
                        help='Drop out rate for hidden layer [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay rate [default: 0.0]')
    parser.add_argument('--wavelet_reg', type=float, default=0.0,
                        help='Regularization term on the wavelet layers [default: 0.0]')
    parser.add_argument('--arch', type=str, default='res',
                        help='Deep arch used [default: resnet, options: fc, conv, res]')
    return parser.parse_args()


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, FLAGS,
                      input_length):
    if classifier_name == 'nne':
        from classifiers import nne
        return nne.Classifier_NNE(output_directory, input_shape,
                                  nb_classes, FLAGS, input_length)
    if classifier_name == 'mix':
        from classifiers import model
        return model.Classifier_MIXTE(output_directory, input_shape, nb_classes, FLAGS=FLAGS, input_length=input_length)


def fit_classifier(classifier_type, data_name):
    parsers = parse_arguments()

    data_dir = "./data"

    X_train, Y_train, X_test, Y_test, Y_true, nb_classes, _ = load_data(data_dir, data_name, parsers.batch_size)

    LEN_INPUT = len(X_train[0])

    input_shape = X_train.shape[1:]

    classifier = create_classifier(classifier_type, input_shape, nb_classes,
                                   output_directory, FLAGS=parsers, input_length=LEN_INPUT)

    classifier.fit(X_train, Y_train, X_test, Y_test, Y_true)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    root_dir = '.'
    xps = ['use_bottleneck', 'use_residual', 'nb_filters', 'depth',
           'kernel_size', 'batch_size']

    # run nb_iter_ iterations of Inception on the whole TSC archive
    classifier_name = 'mix'
    archive_name = ARCHIVE_NAMES[0]
    nb_iter_ = 5

    # datasets_dict = read_all_datasets(data_dir, archive_name)

    for iter in range(nb_iter_):
        print('\t\titer', iter)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)

        tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + trr + '/'

        for dataset_name in dataset_names_for_archive[archive_name]:
            print('\t\t\tdataset_name: ', dataset_name)

            output_directory = tmp_output_directory + dataset_name + '/'

            temp_output_directory = create_directory(output_directory)

            if temp_output_directory is None:
                print('Already_done', tmp_output_directory, dataset_name)
                continue

            fit_classifier(classifier_name, dataset_name)

            print(f'\t\t\t\t{classifier_name} fit DONE')

            # the creation of this directory means
            create_directory(output_directory + '/DONE')

    # run the ensembling of these iterations of Inception
    classifier_name = 'nne'
    print(f"{classifier_name}".center(80, "-"))

    tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'

    for dataset_name in dataset_names_for_archive[archive_name]:
        print('\t\t\tdataset_name: ', dataset_name)

        # x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

        output_directory = tmp_output_directory + dataset_name + '/'

        fit_classifier(classifier_name, dataset_name)

        print('\t\t\t\tDONE')