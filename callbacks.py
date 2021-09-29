import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import BaseLogger
import tensorflow.keras.backend as K

LEARNING_RATE_DECAY = 0.98
LEARING_RATE_DECAY_EVERY_N_EPOCHS = 2

def scheduler(epoch, learning_rate):
    if epoch > 0:
        if epoch % LEARING_RATE_DECAY_EVERY_N_EPOCHS == 0:
            learning_rate = learning_rate*LEARNING_RATE_DECAY
            print('Change learning rate to', "{0:.6f}".format(learning_rate))
    return learning_rate


class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=5, startAt=1):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.intEpoch) % self.every == 0:
            # Save current model weight
            p = os.path.sep.join([self.outputPath, "epoch_{}.h5".format(self.intEpoch)])
            self.model.save_weights(p, overwrite=True)

            # Delete old model weight
            old_p = os.path.sep.join([self.outputPath, "epoch_{}.h5".format(self.intEpoch - self.every)])
            if os.path.exists(old_p):
                os.remove(old_p)
        # increment the internal epoch counter
        self.intEpoch += 1


class SaveLog(BaseLogger):
    def __init__(self, jsonPath=None, jsonName=None, startAt=0, verbose=0):
        super(SaveLog, self).__init__()
        self.jsonPath = jsonPath
        self.jsonName = jsonName
        self.jsonfile = os.path.join(self.jsonPath, self.jsonName)
        self.startAt = startAt
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonfile is not None:
            if os.path.exists(self.jsonfile):
                self.H = json.loads(open(self.jsonfile).read())

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            # if len(v)>1:
            #     v = np.mean(v)
            l.append(float(v))
            self.H[k] = l


        # check to see if the training history should be serialized
        # to file
        if self.jsonfile is not None:
            f = open(self.jsonfile, "w")
            f.write(json.dumps(self.H))
            f.close()


class EpochCheckpoint_Two_Models(Callback):
    def __init__(self, outputPath, every=5, startAt=1):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.intEpoch) % self.every == 0:
            # Save current model weight
            p1 = os.path.sep.join([self.outputPath, "tasnet1_epoch_{}.h5".format(self.intEpoch)])
            self.model.model1.save_weights(p1, overwrite=True)
            p2 = os.path.sep.join([self.outputPath, "tasnet2_epoch_{}.h5".format(self.intEpoch)])
            self.model.model2.save_weights(p2, overwrite=True)

            # Delete old model weight
            old_p1 = os.path.sep.join([self.outputPath, "tasnet1_epoch_{}.h5".format(self.intEpoch - self.every)])
            old_p2 = os.path.sep.join([self.outputPath, "tasnet2_epoch_{}.h5".format(self.intEpoch - self.every)])
            if os.path.exists(old_p1):
                os.remove(old_p1)
            if os.path.exists(old_p2):
                os.remove(old_p2)
        # increment the internal epoch counter
        self.intEpoch += 1

def find_current_lr(lr_init, LR_decay_every_N_epochs, LR_decay_factor, current_epoch):
    lr = lr_init
    for i in range(current_epoch):
        if i%LR_decay_every_N_epochs==0:
            lr = lr* LR_decay_factor
    return lr

def find_current_confidence(c_init, current_epoch, min_c=-20):
    new_c = c_init
    temp = current_epoch // 10
    new_c -= temp
    if new_c< min_c:
        new_c = min_c
    return new_c

class ModelCheckpoint_and_Reduce_LR(BaseLogger):
    def __init__(self, folderpath, jsonPath=None, jsonName=None, startAt=0, monitor='val_accuracy', mode='max', lr_init=0.001,
                 LR_decay_every_N_epochs=2, patience=10, LR_decay_factor=0.98, min_lr=1e-8, current_confidence=-15,
                 n_epoch_add_confidence=10, min_c=-20,
                 verbose=1):
        super(ModelCheckpoint_and_Reduce_LR, self).__init__()
        self.filepath_m1 = os.path.join(folderpath, 'best_m1_model.h5')
        self.filepath_m2 = os.path.join(folderpath, 'best_m2_model.h5')
        self.jsonPath = jsonPath
        self.jsonName = jsonName
        self.jsonfile = os.path.join(self.jsonPath, self.jsonName)
        self.startAt = startAt
        self.intEpoch = startAt
        self.monitor = monitor
        self.mode = mode
        self.LR_decay_factor = LR_decay_factor
        self.count_epoch = 0
        self.lr_init = lr_init
        self.min_c = min_c
        self.current_confidence = current_confidence
        self.LR_decay_every_N_epochs = LR_decay_every_N_epochs
        self.n_epoch_add_confidence = n_epoch_add_confidence
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose

        if self.mode == 'max':
            self.monitor_op = np.greater
            self.current_best = -np.Inf
            self.current_best_M1 = -np.Inf
            self.current_best_M2 = -np.Inf
        else:
            self.monitor_op = np.less
            self.current_best = np.Inf
            self.current_best_M1 = np.Inf
            self.current_best_M2 = np.Inf

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}
        # Find current lr and set to model
        lr = self.lr_init
        current_confidence = self.current_confidence
        if self.startAt>1:
            lr = find_current_lr(self.lr_init, self.LR_decay_every_N_epochs, self.LR_decay_factor, self.startAt)
            print('Current LR: ', lr)
            current_confidence = find_current_confidence(-15, self.startAt, min_c=self.min_c)
        K.set_value(self.model.optimizer.lr, lr)
        self.model.confidence = current_confidence
        print('Current current_confidence: ', current_confidence)

        # if the JSON history path exists, load the training history
        if self.jsonfile is not None:
            if os.path.exists(self.jsonfile)  and self.startAt > 1:
                self.H = json.loads(open(self.jsonfile).read())
                if self.mode == 'max':
                    self.current_best = max(self.H[self.monitor])
                    self.current_best_M1 = max(self.H['val_loss_M1'])
                    self.current_best_M2 = max(self.H['val_loss_M2'])
                    self.idx = self.H[self.monitor].index(self.current_best)
                else:
                    self.current_best = min(self.H[self.monitor])
                    self.current_best_M1 = min(self.H['val_loss_M1'])
                    self.current_best_M2 = min(self.H['val_loss_M2'])
                    self.idx = self.H[self.monitor].index(self.current_best)
                for k in self.H.keys():
                    self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        current = logs.get(self.monitor)
        if len(current)>1:
            current = np.mean(current)
        lr = K.get_value(self.model.optimizer.lr)
        print('\nCurrent best loss: ', self.current_best)
        print('Current LR: %0.5f and count epoch: %03d.' % (lr, self.count_epoch))
        self.count_epoch += 1
        self.intEpoch += 1

        if self.monitor_op(current, self.current_best):
            if self.verbose > 0:
                print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s'
                      % (epoch + 1, self.monitor, self.current_best, current, self.filepath_m1))

            # self.model.model1.save_weights(self.filepath_m1, overwrite=True)
            # self.model.model2.save_weights(self.filepath_m2, overwrite=True)
            self.current_best = current
            self.count_epoch = 0

        # Save model weights of M1
        current = logs.get('val_loss_M1')
        if len(current)>1:
            current = np.mean(current)
        if self.monitor_op(current, self.current_best_M1):
            self.model.model1.save_weights(self.filepath_m1, overwrite=True)
            self.current_best_M1 = current
            print('Saving model weights of model 1 with new best loss %0.5f'%(self.current_best_M1))

        # Save model weights of M2
        current = logs.get('val_loss_M2')
        if len(current) > 1:
            current = np.mean(current)
        if self.monitor_op(current, self.current_best_M2):
            self.model.model2.save_weights(self.filepath_m2, overwrite=True)
            self.current_best_M2 = current
            print('Saving model weights of model 2 with new best loss %0.5f' % (self.current_best_M2))

        # Early Stopping
        if self.count_epoch >= self.patience:
            print('Epoch %05d: early stopping' % (epoch + self.startAt))
            self.model.stop_training = True

        # Reduce LR
        if self.intEpoch % self.LR_decay_every_N_epochs == 0:
            new_lr = lr * self.LR_decay_factor
            if new_lr < self.min_lr:
                new_lr = self.min_lr
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                print('Epoch %05d: LearningRateScheduler reducing learning '
                      'rate to %s.' % (epoch + self.startAt, new_lr))

        # Add confidence
        if self.intEpoch % self.n_epoch_add_confidence==0:
            if self.model.confidence > self.min_c:
                self.model.confidence -= 1

        for (k, v) in logs.items():
            l = self.H.get(k, [])
            if len(v)>1:
                v = np.mean(v)
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonfile is not None:
            f = open(self.jsonfile, "w")
            f.write(json.dumps(self.H))
            f.close()

#----------------------------------------------------------------------------------------------------------------------------------


class EpochCheckpoint_Teacher_Student_Models(Callback):
    def __init__(self, outputPath, every=5, startAt=1):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.intEpoch) % self.every == 0:
            # Save current model weight
            p_teacher = os.path.sep.join([self.outputPath, "teacher_epoch_{}.h5".format(self.intEpoch)])
            self.model.teacher.save_weights(p_teacher, overwrite=True)
            p_student = os.path.sep.join([self.outputPath, "student_epoch__{}.h5".format(self.intEpoch)])
            self.model.student.save_weights(p_student, overwrite=True)

            # Delete old model weight
            old_p_teacher = os.path.sep.join([self.outputPath, "teacher_epoch_{}.h5".format(self.intEpoch - self.every)])
            old_p_student = os.path.sep.join([self.outputPath, "student_epoch__{}.h5".format(self.intEpoch - self.every)])
            if os.path.exists(old_p_teacher):
                os.remove(old_p_teacher)
            if os.path.exists(old_p_student):
                os.remove(old_p_student)
        # increment the internal epoch counter
        self.intEpoch += 1

class ModelCheckpoint_and_Reduce_LR_Teacher_Student(BaseLogger):
    def __init__(self, folderpath, jsonPath=None, jsonName=None, startAt=0, monitor='val_accuracy', mode='max', lr_init=0.001,
                 LR_decay_every_N_epochs=2, patience=10, LR_decay_factor=0.98, min_lr=1e-8,
                 verbose=1):
        super(ModelCheckpoint_and_Reduce_LR_Teacher_Student, self).__init__()
        self.filepath_teacher = os.path.join(folderpath, 'best_teacher_model.h5')
        self.filepath_student = os.path.join(folderpath, 'best_student_model.h5')
        self.jsonPath = jsonPath
        self.jsonName = jsonName
        self.jsonfile = os.path.join(self.jsonPath, self.jsonName)
        self.startAt = startAt
        self.intEpoch = startAt
        self.monitor = monitor
        self.mode = mode
        self.LR_decay_factor = LR_decay_factor
        self.count_epoch = 0
        self.lr_init = lr_init
        self.LR_decay_every_N_epochs = LR_decay_every_N_epochs
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose

        if self.mode == 'max':
            self.monitor_op = np.greater
            self.current_best = -np.Inf
        else:
            self.monitor_op = np.less
            self.current_best = np.Inf

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}
        # Find current lr and set to model
        lr = self.lr_init
        if self.startAt>1:
            lr = find_current_lr(self.lr_init, self.LR_decay_every_N_epochs, self.LR_decay_factor, self.startAt)
            print('Current LR: ', lr)
        K.set_value(self.model.optimizer.lr, lr)

        # if the JSON history path exists, load the training history
        if self.jsonfile is not None:
            if os.path.exists(self.jsonfile)  and self.startAt > 1:
                self.H = json.loads(open(self.jsonfile).read())
                if self.mode == 'max':
                    self.current_best = max(self.H[self.monitor])
                    self.idx = self.H[self.monitor].index(self.current_best)
                else:
                    self.current_best = min(self.H[self.monitor])
                    self.idx = self.H[self.monitor].index(self.current_best)
                for k in self.H.keys():
                    self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        current = logs.get(self.monitor)
        if len(current)>1:
            current = np.mean(current)
        lr = K.get_value(self.model.optimizer.lr)
        print('\nCurrent best loss: ', self.current_best)
        print('Current LR: %0.5f and count epoch: %03d.' % (lr, self.count_epoch))
        self.count_epoch += 1
        self.intEpoch += 1

        if self.monitor_op(current, self.current_best):
            if self.verbose > 0:
                print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s'
                      % (epoch + 1, self.monitor, self.current_best, current, self.filepath_student))

            self.model.student.save_weights(self.filepath_student, overwrite=True)
            self.current_best = current
            self.count_epoch = 0

        # Early Stopping
        if self.count_epoch >= self.patience:
            print('Epoch %05d: early stopping' % (epoch + self.startAt))
            self.model.stop_training = True

        # Reduce LR
        if self.intEpoch % self.LR_decay_every_N_epochs == 0:
            new_lr = lr * self.LR_decay_factor
            if new_lr < self.min_lr:
                new_lr = self.min_lr
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                print('Epoch %05d: LearningRateScheduler reducing learning '
                      'rate to %s.' % (epoch + self.startAt, new_lr))

        for (k, v) in logs.items():
            l = self.H.get(k, [])
            if len(v)>1:
                v = np.mean(v)
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonfile is not None:
            f = open(self.jsonfile, "w")
            f.write(json.dumps(self.H))
            f.close()