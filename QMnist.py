import os
import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def use_cpu(cpu=True):
    if cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_data():
    print("Loading MNIST data...")
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the data
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def load_client_data(all_clients=False, archi="dnn"):
    print("Loading client data ...")
    with open("data/MNISTX.txt", "rb") as fp:  # Pickling
        X = pickle.load(fp)
        if all_clients:
            train_X = np.vstack(X)
        else:
            train_X = np.array(X[2])
        # CNN
        if archi == "cnn":
            train_X = train_X.reshape(-1, 28, 28, 1)

    with open("data/MNISTy.txt", "rb") as fp:  # Pickling
        y = pickle.load(fp)
        if all_clients:
            train_y = np.vstack(y)
        else:
            train_y = np.array(y[2])
            print(train_y.shape)

    return (train_X, train_y), (train_X, train_y)


def build_model(shape=784, classes=10):
    print("Building model...")
    dnn = Sequential()
    dnn.add(Dense(200, input_shape=(shape,)))
    dnn.add(Activation("sigmoid"))
    dnn.add(Dense(200))
    dnn.add(Activation("relu"))
    dnn.add(Dense(classes))
    dnn.add(Activation("softmax"))
    return dnn


def build_cnn_model():
    print("Building model...")
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    print(cnn.summary())
    exit()
    return cnn
    print("Building model...")
    # Load ResNet50 with pre-trained ImageNet weights, without the top layer
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Add new layers for MNIST classification
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # Create the model
    resnet50 = Model(inputs=base.input, outputs=predictions)

    return resnet50


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
        qmodel.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                   validation_data=(test_images, test_labels),
                   verbose=3)
        # simulate network activity
        time.sleep(0.2)


def test_model(test_dataset):
    print("Testing model...")
    test_images, test_labels = test_dataset
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)


if __name__ == '__main__':
    start = time.time()
    R = 10
    num_bits = 32
    print(f"Device started FL training for {R} rounds with {num_bits} quantization:")
    use_cpu(True)
    # train_data, test_data = load_data()
    train_data, test_data = load_client_data(archi="dnn")
    model = build_model()
    if num_bits in [8, 16]:
        optimizer = QuantizedSGD(learning_rate=0.01, quantization_bits=num_bits)
    else:
        optimizer = 'adam'
    federated_learning(model, optimizer, train_data, test_data, epochs=10, rounds=R)
    # test_model(test_data)
    print("Done in {} seconds".format(round(time.time() - start, 4)))
