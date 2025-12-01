import tensorflow as tf
from tensorflow import keras
import numpy as np

class FashionMNISTCNN:
    def __init__(self):
        inputs = keras.Input(shape=(28, 28, 1))
        x = keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
        x = keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def load_and_preprocess_data(self):
        # Загрузка и нормализация данных Fashion-MNIST
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        # Добавление каналовой размерности
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        return X_train, y_train, X_test, y_test

    def create_data_augmentation(self):
        # Pipeline аугментации данных
        return keras.Sequential([
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomTranslation(0.1, 0.1)
        ])

    def compile_model(self):
        # Компиляция модели
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        # Обучение модели с аугментацией
        data_aug = self.create_data_augmentation()
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(10000).batch(batch_size)
        train_ds = train_ds.map(lambda x, y: (data_aug(x, training=True), y))
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        self.model.fit(train_ds, validation_split=0.2, epochs=epochs, verbose=2)

    def evaluate(self, X_test, y_test):
        # Оценка на тестовой выборке
        return self.model.evaluate(X_test, y_test, verbose=0)

# Пример использования
cnn = FashionMNISTCNN()
X_train, y_train, X_test, y_test = cnn.load_and_preprocess_data()
cnn.compile_model()
cnn.train(X_train, y_train, epochs=5, batch_size=64)
test_loss, test_acc = cnn.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")