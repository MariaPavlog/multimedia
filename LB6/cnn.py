import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# Преобразование меток в категориальный формат
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Создание модели CNN
cnn_model = models.Sequential()
#сверточный (32 фильтра размера 3 на 3)
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#подвыборочный
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# преобразование в одномерный массив для полносвязных слоев
cnn_model.add(layers.Flatten())
#полносвязные
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(layers.Dense(10, activation='softmax'))


# Компиляция модели
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

epochs_list = [1, 3, 7, 10]
for epochs in epochs_list:
    print(f'\nTraining for {epochs} epochs:')
    cnn_model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=epochs, batch_size=32, validation_split=0.2)
    test_loss, test_acc = cnn_model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)
    print(f'Test accuracy: {test_acc:.4f}')

