import time
import keras
import numpy as np
import tensorflow as tf
from keras.regularizers import l2
from .network import get_wave_kernel, wave_variable_with_l1, variable_on_cpu, tf_concat
from utils import *


class Classifier_MIXTE:
    def __init__(self, output_directory, input_shape, nb_classes, FLAGS, input_length, verbose=False, build=True,
                 batch_size=28, nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):
        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = FLAGS.batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.l1_value = 0.000
        self.use_bn = True
        self.SIM_REG = FLAGS.wavelet_reg
        self.mid_neuron_num_1 = 40
        self.dp_keep_prob_input = 1 - FLAGS.drop_rate_input
        self.dp_keep_prob_hidden = 1 - FLAGS.drop_rate_hidden
        self.weight_decay_conv = FLAGS.weight_decay
        self.weight_decay_fc = FLAGS.weight_decay
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.input_length = input_length
        if build:
            self.model = self.build_model()
            if verbose:
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    @staticmethod
    def _shortcut_layer(input_res_tensor, input_tensor):
        print(input_tensor.shape[-1])
        shortcut_y = keras.layers.Conv1D(filters=int(input_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_res_tensor)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, input_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def res_block(self, input_data, num_channel_out, ker_size, bn, weight_decay):
        print(f"res block".center(40, '-'))
        num_channel_in = input_data.get_shape()[-1]
        conv1 = self.conv2d(input_data, bn, ker_size, num_channel_out, weight_decay)
        if num_channel_out != num_channel_in:
            side_conv = self.conv2d(input_data, bn, 1, num_channel_out, weight_decay)
        else:
            side_conv = input_data
        output = keras.layers.Add()([conv1, side_conv])
        return output

    @staticmethod
    def conv2d(inputs, bn, ker_size, num_channel_out, weight_decay):
        # num_channel_in = inputs.get_shape()[-1].value
        # print(num_channel_in)
        conv2 = keras.layers.Conv2D(num_channel_out,
                                    ker_size,
                                    padding='same',
                                    activation='relu',
                                    kernel_regularizer=l2(weight_decay))(inputs)
        if bn:
            conv2 = keras.layers.normalization.BatchNormalization()(conv2)
        return conv2

    @staticmethod
    def avg_pool2d(input, kernel_size, stride=None, padding='valid'):
        print(kernel_size)
        print(input.shape)
        if stride is None:
            stride = [2, 2]
        outputs = keras.layers.AveragePooling2D(pool_size=kernel_size, strides=stride, padding=padding)(input)
        return outputs

    @staticmethod
    def max_pool2d(input, kernel_size, stride=None, padding='valid'):
        if stride is None:
            stride = [2, 2]
        outputs = keras.layers.MaxPooling2D(pool_size=kernel_size, strides=stride, padding=padding)(input)
        return outputs

    @staticmethod
    def fully_connected(inputs,
                        num_outputs,
                        use_xavier=True,
                        stddev=1e-2,
                        weigth_decay=0.0,
                        activation_fn='relu',
                        bn=True):
        if use_xavier:
            initializer = keras.initializers.glorot_normal()
        else:
            initializer = keras.initializers.RandomNormal(stddev=stddev)
        outputs = keras.layers.Dense(num_outputs, activation=activation_fn,
                                     kernel_regularizer=keras.regularizers.l2(weigth_decay),
                                     kernel_initializer=initializer,
                                     bias_initializer=keras.initializers.Zeros())(inputs)
        if bn:
            outputs = keras.layers.BatchNormalization()(outputs)
        print('fully connect layer')
        return outputs

    @staticmethod
    def wave_op_conv(input, len_input, l1_value, weight_decay, sim_reg=0, activation=None):
        input = tf.squeeze(input)
        lp_mat, hp_mat = get_wave_kernel([len_input, len_input / 2])
        lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=weight_decay, l1_value=l1_value, sim_reg=sim_reg)
        hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=weight_decay, l1_value=l1_value, sim_reg=sim_reg)
        biases_lp = variable_on_cpu('biases_lp_' + str(int(len_input / 2)), [int(len_input / 2)],
                                    tf.constant_initializer(0.0))
        biases_hp = variable_on_cpu('biases_hp' + str(int(len_input / 2)), [int(len_input / 2)],
                                    tf.constant_initializer(0.0))
        lp_out = tf.matmul(input, lp_weight)
        lp_out = tf.nn.bias_add(lp_out, biases_lp)

        hp_out = tf.matmul(input, hp_weight)
        hp_out = tf.nn.bias_add(hp_out, biases_hp)

        if not activation == None:
            hp_out = activation(hp_out)
            lp_out = activation(lp_out)

        hp_out = tf.expand_dims(hp_out, -1)
        lp_out = tf.expand_dims(lp_out, -1)

        hp_out = tf.expand_dims(hp_out, -1)
        lp_out = tf.expand_dims(lp_out, -1)

        all_out = tf_concat(-1, [lp_out, hp_out])
        return lp_out, hp_out, all_out

    def wave_block_res(self, input, num_outputs, len_input):
        lp_coe, hp_coe, all_coe = self.wave_op_conv(input, len_input, l1_value=self.l1_value,
                                                    weight_decay=self.weight_decay_fc,
                                                    sim_reg=self.SIM_REG)

        conv1 = self.res_block(all_coe, 8, 8, bn=self.use_bn, weight_decay=self.weight_decay_conv)
        conv1 = self.res_block(conv1, 8, 8, bn=self.use_bn, weight_decay=self.weight_decay_conv)
        conv1 = self.res_block(conv1, 8, 8, bn=self.use_bn, weight_decay=self.weight_decay_conv)
        conv1 = self.max_pool2d(conv1, [2, 1], stride=[2, 1])

        conv2 = self.res_block(conv1, 16, 5, bn=self.use_bn, weight_decay=self.weight_decay_conv)
        conv2 = self.res_block(conv2, 16, 5, bn=self.use_bn, weight_decay=self.weight_decay_conv)
        conv2 = self.res_block(conv2, 16, 5, bn=self.use_bn, weight_decay=self.weight_decay_conv)
        conv2 = self.max_pool2d(conv2, [2, 1], stride=[2, 1])

        conv3 = self.res_block(conv2, 32, 3, bn=self.use_bn, weight_decay=self.weight_decay_conv)
        conv3 = self.res_block(conv3, 32, 3, bn=self.use_bn, weight_decay=self.weight_decay_conv)
        conv3 = self.res_block(conv3, 32, 3, bn=self.use_bn, weight_decay=self.weight_decay_conv)

        len_conv = conv3.get_shape()[1].value
        conv3 = self.avg_pool2d(conv3, [len_conv, 1], stride=[1, 1])
        conv3 = tf.squeeze(conv3)
        predict = self.fully_connected(conv3, num_outputs, bn=False)
        return lp_coe, predict

    def _input_layer(self, batch_size=None):
        if batch_size is None:
            return keras.layers.Input(self.input_shape)
        else:
            return keras.layers.Input(batch_shape=(batch_size, self.input_length))

    def _mwd_layer(self, input_tensor):

        x = input_tensor

        lp_1, predict_1 = self.wave_block_res(x, self.nb_classes, self.input_length)

        lp_2, predict_2 = self.wave_block_res(lp_1, self.nb_classes, self.input_length / 2)

        predict_2 = keras.layers.Add()([predict_1, predict_2])
        # predict_2 = tf.add(predict_2, predict_1, name='adding_2')

        lp_3, predict_3 = self.wave_block_res(lp_2, self.nb_classes, self.input_length / 4)
        # predict_3 = tf.add(predict_3, predict_2, name='adding_3')
        predict_3 = keras.layers.Add()([predict_3, predict_2])

        lp_4, predict_4 = self.wave_block_res(lp_3, self.nb_classes, self.input_length / 8)

        predict_4 = keras.layers.Add()([predict_3, predict_4])
        # predict_4 = tf.add(predict_4, predict_3, name='adding_4')
        return predict_4

    def _inceptionTime_layer(self, input_tensor):
        x = tf.expand_dims(input_tensor, -1)
        input_res = x

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)
        # output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        return gap_layer

    def _concatenate_layer(self, input_layer):
        inception_tensor = self._inceptionTime_layer(input_layer)
        mwd_tensor = self._mwd_layer(input_layer)
        return tf.concat([inception_tensor, mwd_tensor], axis=1)

    def build_model(self):
        input_layer = self._input_layer(batch_size=self.batch_size)

        # inception_tensor = keras.layers.Lambda(self._inceptionTime_layer, name="inception_tensor")(input_layer)
        # mwd_tensor = keras.layers.Lambda(self._mwd_layer, name="mwd_tensor")(input_layer)

        concatenate_tensor = keras.layers.Lambda(self._concatenate_layer, name="concatenate_tensor")(input_layer)
        # output_layer = keras.layers.concatenate([inception_tensor, mwd_tensor])
        output_layer = keras.layers.Dense(self.nb_classes, activation='softmax')(concatenate_tensor)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        x_test = x_test.squeeze()
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        print(f"model_path: {model_path}".center(80, '*'))
        model = keras.models.load_model(model_path,
                                        custom_objects={
                                            '_concatenate_layer': self._concatenate_layer,
                                            '_inceptionTime_layer': self._inceptionTime_layer,
                                            '_mwd_layer': self._mwd_layer
                                        })
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred

    def fit(self, x_train, y_train, x_val, y_val, y_true, plot_test_acc=False):

        x_train = x_train.squeeze()
        y_train = y_train.squeeze()

        print(f'begin fit'.center(80, '*'))
        # print(x_train.shape)
        # if not tf.test.is_gpu_available():
        #     print('error no gpu')
        #     exit()
        # else:
        #     print(f"begin fit".center(80, '-'))
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size
        print(f"mini batch size: {mini_batch_size}".center(80, '-'))
        start_time = time.time()

        if plot_test_acc:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        # print(f"predict the result".center(50, '-'))
        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # save predictions
        # print(f"save predictions".center(50, '-'))
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration,
                               plot_test_acc=plot_test_acc)

        keras.backend.clear_session()

        print(f"Finished fit".center(80, '*'))
        return df_metrics
