# Script for ResNet model on our ECG data. Trains and tests the model.
"""
Look at the output of the script. It now shows plots of the accuracy, loss and a confusion matrix. We should change this
depending on what we want to analyse. With that, we can make our own plots for the comparison.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow_addons.optimizers import CyclicalLearningRate
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import sklearn
import itertools

# Load train and test data
path_train = "./physionet_apnea/a01.csv" # SET CORRECTLY BEFORE RUNNING
path_test = "./physionet_apnea/b04.csv" # SET CORRECTLY BEFORE RUNNING


df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

train_arr = np.array(df_train)
test_arr = np.array(df_test)

X_train = train_arr[:,1:]
X_test = test_arr[:,1:]
y_train = train_arr[:,0]
y_test = test_arr[:,0]

# Split test in validation and test set
X_val, X_test, y_val, y_test  = train_test_split(X_testval, y_testval, test_size=0.5, random_state=42)


# Function to plot loss and accuracy
def pretty_plot(history, field, fn):
    def plot(data, val_data, best_index, best_value, title):
        plt.plot(range(1, len(data) + 1), data, label='train')
        plt.plot(range(1, len(data) + 1), val_data, label='validation')
        if not best_index is None:
            plt.axvline(x=best_index + 1, linestyle=':', c="#777777")
        if not best_value is None:
            plt.axhline(y=best_value, linestyle=':', c="#777777")
        plt.xlabel('Epoch')
        plt.ylabel(field)
        plt.xticks(range(0, len(data), 20))
        plt.title(title)
        plt.legend()
        plt.show()

    data = history.history[field]
    val_data = history.history['val_' + field]
    tail = int(0.15 * len(data))

    best_index = fn(val_data)
    best_value = val_data[best_index]

    plot(data, val_data, best_index, best_value, "{} over epochs (best {:06.4f})".format(field, best_value))
    plot(data[-tail:], val_data[-tail:], None, best_value, "{} over last {} epochs".format(field, tail))


# Function to create ResNet model
def get_resnet_model(categories=2):
    def residual_block(X, kernels, stride):
        out = keras.layers.Conv1D(kernels, stride, padding='same')(X)
        out = keras.layers.ReLU()(out)
        out = keras.layers.Conv1D(kernels, stride, padding='same')(out)
        out = keras.layers.add([X, out])
        out = keras.layers.ReLU()(out)
        out = keras.layers.MaxPool1D(5, 2)(out)
        return out

    kernels = 32
    stride = 5

    inputs = keras.layers.Input([6000, 1])
    X = keras.layers.Conv1D(kernels, stride)(inputs)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(32, activation='relu')(X)
    X = keras.layers.Dense(32, activation='relu')(X)
    output = (keras.layers.Dense(1, activation='sigmoid')(X) if categories == 2 else keras.layers.Dense(5,
                                                                                                        activation='softmax')(
        X))

    model = keras.Model(inputs=inputs, outputs=output)
    return model



# Copied from https://github.com/avanwyk/tensorflow-projects/blob/master/lr-finder/lr_finder.py
# Apache License 2.0
class LRFinder(Callback):
    """`Callback` that exponentially adjusts the learning rate after each training batch between `start_lr` and
    `end_lr` for a maximum number of batches: `max_step`. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the `plot` method.
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 100, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# "Pretrain" the model for finding good learning rate(s)
model = get_resnet_model()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
lr_finder = LRFinder(start_lr=1e-7, end_lr= 1e-03, max_steps=100, smoothing=0.6)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=256, epochs=5, callbacks=[lr_finder], verbose=False)
lr_finder.plot()

# Set cyclical learning rate
N = X_train.shape[0]
batch_size = 128
iterations = N/batch_size
step_size= 2 * iterations

# Get learning rate planning
lr_schedule = CyclicalLearningRate(1e-6, 1e-3, step_size=step_size, scale_fn=lambda x: tf.pow(0.95,x))
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

save_best_weights = ModelCheckpoint(filepath="weights.hdf5", verbose=0, save_best_only=True)

# Create and train new ResNet model
resnet_model = get_resnet_model()
resnet_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = resnet_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                           shuffle=True, batch_size=batch_size, epochs=75, callbacks=[save_best_weights])

# Plot the loss and accuracy over the train and validation set
pretty_plot(history, 'loss', lambda x: np.argmin(x))
pretty_plot(history, 'accuracy', lambda x: np.argmax(x))

# Evaluate the model on test data
resnet_model.load_weights('weights.hdf5')
resnet_model.evaluate(X_test, y_test) # returns loss and accuracy on the test data
y_pred = (resnet_model.predict(X_test) > 0.5).astype("int32")
# Compute confusion matrix
cnf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'],
                      title='Confusion matrix, without normalization')