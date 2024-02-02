import os
import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


def use_cpu(cpu=True):
    if cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_data():
    print("Loading MNIST data...")
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def load_client_data(all_clients=False):
    print("Loading client data ...")
    with open("data/CIFARX.txt", "rb") as fp:  # Pickling
        X = pickle.load(fp)
        if all_clients:
            train_X = np.vstack(X)
            train_X = train_X.reshape(-1, 32, 32, 3)
        else:
            train_X = np.array(X[2]).reshape(-1, 32, 32, 3)

    with open("data/CIFARy.txt", "rb") as fp:  # Pickling
        y = pickle.load(fp)
        if all_clients:
            train_y = np.vstack(y)
        else:
            train_y = np.array(y[2])

    return (train_X, train_y), (train_X, train_y)


def build_model(shape=3072, classes=10):
    print("Building model...")
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return cnn
    # dnn = Sequential()
    # # cifar
    # dnn.add(Flatten(input_shape=(32, 32, 3)))
    # dnn.add(Dense(200))
    # # local cifar
    # # dnn.add(Dense(200, input_shape=(shape,)))
    #
    # dnn.add(Activation("sigmoid"))
    # dnn.add(Dense(200))
    # dnn.add(Activation("relu"))
    # dnn.add(Dense(classes))
    # dnn.add(Activation("softmax"))
    # print(dnn.summary())
    # exit()
    # return dnn


class QuantizedSGD(tf.keras.optimizers.SGD):
    def __init__(self, learning_rate=0.01, quantization_bits=8, **kwargs):
        super(QuantizedSGD, self).__init__(learning_rate, **kwargs)
        self.quantization_bits = quantization_bits

    def _resource_apply_dense(self, grad, var, apply_state):
        if self.quantization_bits == 8 or self.quantization_bits == 16:
            # Quantize the gradient
            quantized_grad = tf.quantization.fake_quant_with_min_max_vars(
                grad, min=-1, max=1, num_bits=self.quantization_bits
            )
        elif self.quantization_bits == 32:
            # Use 32-bit float (standard precision) - no quantization
            quantized_grad = tf.cast(grad, tf.float32)
        elif self.quantization_bits == 64:
            # Use 64-bit float (double precision) - no quantization
            quantized_grad = tf.cast(grad, tf.float64)
        else:
            raise ValueError("Unsupported number of bits for quantization. Supported: 8, 16, 32, 64.")

        return super(QuantizedSGD, self)._resource_apply_dense(quantized_grad, var, apply_state)


def federated_learning(qmodel, opt, train_dataset, test_dataset, epochs=10, rounds=100):
    batch_size = 64
    print(f"Federated learning training with batch_size={batch_size}")
    train_images, train_labels = train_dataset
    test_images, test_labels = test_dataset
    qmodel.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    for r in range(rounds):
        print("Running round {}".format(r + 1), end='\r')
        qmodel.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels),
                   verbose=0)
        # simulate network activity
        time.sleep(0.2)


def test_model(test_dataset):
    print("Testing model...")
    test_images, test_labels = test_dataset
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)


if __name__ == '__main__':
    start = time.time()
    R = 5
    num_bits = 32
    print(f"Device started FL training for {R} rounds with {num_bits} quantization:")
    use_cpu(True)
    # train_data, test_data = load_data()
    train_data, test_data = load_client_data()
    model = build_model()
    if num_bits in [8, 16]:
        optimizer = QuantizedSGD(learning_rate=0.01, quantization_bits=num_bits)
    else:
        optimizer = 'adam'
    federated_learning(model, optimizer, train_data, test_data, epochs=10, rounds=R)
    # test_model(test_data)
    print("Done in {} seconds".format(round(time.time() - start, 4)))
