import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import cv2


# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# Преобразование меток в категориальный формат
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
#print(y_test)

# Создание модели MLP
model = Sequential()

model.add(Dense(128, input_shape=(28 * 28,), activation='relu'))   # Скрытый слой с 128 нейронами
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Выходной слой для 10 классов
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#метрика accuracy для оценки производительности
#функция потерь categorical_crossentropy
#измеряет разницу между двумя распределениями вероятностей:
#предсказанным распределением вероятностей и истинным распределением


model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
test_loss, test_acc = model.evaluate(x_test, y_test)
predictions = model.predict(x_test)
test_loss1, test_accuracy1 = model.evaluate(x_test, y_test)
print(f'Test accuracy 5 epochs: {test_accuracy1}')
# Обучение с разным количеством эпох


for i in range(100,107):
    # Преобразуем массив в изображение
    img = (x_test[i] * 255).astype(np.uint8).reshape(28, 28)

    # Увеличиваем изображение (например, до 280x280)
    img_resized = cv2.resize(img, (280, 280))

    # Добавляем текст на увеличенном изображении
    cv2.putText(img_resized, f'Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(y_test[i])}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    # Отображаем изображение
    cv2.imshow('Image', img_resized)
    cv2.waitKey(0)
cv2.destroyAllWindows()


#№2
epochs_list = [1, 3, 7, 10]
for epochs in epochs_list:
    print(f'\nTraining for {epochs} epochs:')
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')
