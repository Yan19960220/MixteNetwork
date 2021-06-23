# -*- coding: utf-8 -*-
# @Time    : 4/15/21 6:28 PM
# @Author  : Yan
# @Site    :
# @File    : nne.py
# @Software: PyCharm

import keras
import numpy as np
import gc
import time

from utils import *


class Classifier_NNE:
    def create_classifier(self, model_name, input_shape, nb_classes, output_directory, flags, input_length, verbose=False,
                          build=True):
        if self.check_if_match('mix*', model_name):
            from classifiers import model
            return model.Classifier_MIXTE(output_directory, input_shape, nb_classes, flags, input_length,
                                                  build=build)

    def check_if_match(self, rex, name2):
        import re
        pattern = re.compile(rex)
        return pattern.match(name2)

    def __init__(self, output_directory, input_shape, nb_classes, flags, input_length, verbose=False, nb_iterations=5,
                 clf_name='mix'):
        self.classifiers = [clf_name]
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.input_length = input_length
        self.flags = flags
        out_add = ''
        for cc in self.classifiers:
            out_add = out_add + cc + '-'
        self.archive_name = ARCHIVE_NAMES[0]
        self.iterations_to_take = [i for i in range(nb_iterations)]
        for cc in self.iterations_to_take:
            out_add = out_add + str(cc) + '-'
        self.output_directory = output_directory.replace('nne',
                                                         'nne' + '/' + out_add)
        create_directory(self.output_directory)
        self.dataset_name = output_directory.split('/')[-2]
        self.verbose = verbose
        self.models_dir = output_directory.replace('nne', 'classifier')

    def fit(self, x_train, y_train, x_test, y_test, y_true):
        # no training since models are pre-trained
        start_time = time.time()

        y_pred = np.zeros(shape=y_test.shape)

        ll = 0

        # loop through all classifiers
        for model_name in self.classifiers:
            # loop through different initialization of classifiers
            for itr in self.iterations_to_take:
                if itr == 0:
                    itr_str = ''
                else:
                    itr_str = '_itr_' + str(itr)

                curr_archive_name = self.archive_name + itr_str

                curr_dir = self.models_dir.replace('classifier', model_name).replace(
                    self.archive_name, curr_archive_name)

                model = self.create_classifier(model_name, self.input_shape, self.nb_classes,
                                               curr_dir, flags=self.flags, input_length=self.input_length, build=False)

                predictions_file_name = curr_dir + 'y_pred.npy'
                # check if predictions already made
                if check_if_file_exits(predictions_file_name):
                    # then load only the predictions from the file
                    curr_y_pred = np.load(predictions_file_name)
                else:
                    # then compute the predictions
                    curr_y_pred = model.predict(x_test, y_true, x_train, y_train, y_test,
                                                return_df_metrics=False)

                    np.save(predictions_file_name, curr_y_pred)

                    keras.backend.clear_session()

                y_pred = y_pred + curr_y_pred

                ll += 1

        # average predictions
        y_pred = y_pred / ll

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        duration = time.time() - start_time

        df_metrics = calculate_metrics(y_true, y_pred, duration)

        df_metrics.to_csv(self.output_directory + 'df_metrics.csv', index=False)

        gc.collect()
