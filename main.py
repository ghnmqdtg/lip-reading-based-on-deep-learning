import os
from lipreading.model import Lipreading
from utils.dataset import DataLoader
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model_training(verbose=True):
    train_set, test_set = DataLoader().load_data()
    input_shape = (120, 96, 96, 1)
    model = Lipreading(input_shape)

    if not config.TRAINING_FROM_SCRATCH:
        try:
            model = tf.keras.models.load_model(config.PATH_WEIGHTS)
        except:
            pass

    # model.build((5, 120, 96, 96, 1))
    # model.summary()

    optim = Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-07, decay=0.0, amsgrad=True, name='Adam')
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['acc'])

    # Set up EarlyStopping and checkpoint
    earlytopping = EarlyStopping(monitor='loss', patience=10)
    checkpoint = ModelCheckpoint(filepath=config.PATH_WEIGHTS,
                                 monitor='acc',
                                 verbose=verbose,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto')

    history = model.fit(train_set,
                        validation_split=0,
                        epochs=config.EPOCH_NUM,
                        batch_size=config.BATCH_SIZE,
                        verbose=1 if verbose else 0,
                        callbacks=[checkpoint],
                        validation_data=test_set)

    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    acc = history.history.get('acc')
    val_acc = history.history.get('val_acc')

    plt.figure(0)
    plt.subplot(121)
    plt.plot(range(len(loss)), loss, label='Training')
    plt.plot(range(len(val_loss)), val_loss, label='Validation')
    plt.title('MSE')
    plt.legend(loc='best')
    plt.subplot(122)
    plt.plot(range(len(acc)), acc, label='Training')
    plt.plot(range(len(val_acc)), val_acc, label='Validation')
    plt.title('ACC')
    plt.legend(loc='best')
    plt.savefig(
        config.PATH_CURVE, dpi=300, format='png')
    plt.close()
    print(
        f'Result saved into {config.PATH_CURVE}')
    
    if verbose:
        plt.show()


def model_validation(verbose=True):
    train_set, test_set = DataLoader().load_data()
    train_set_contents = list(train_set.as_numpy_iterator())

    model = tf.keras.models.load_model(config.PATH_WEIGHTS)
    model.evaluate(train_set, verbose=2)

if __name__ == '__main__':
    # model_training(verbose=True)
    model_validation()
