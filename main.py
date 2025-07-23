import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

def main():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog','frog','horse','ship','truck']

    plt.figure(figsize=(10,5))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i][0]])
    plt.suptitle("Örnek Eğitim Görselleri", fontsize = 14)
    plt.show()

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0 

    datagen = ImageDataGenerator(
        rotation_range = 15,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True
    )
    datagen.fit(train_images)

    def build_model():
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            layers.Conv2D(32, (3, 3), activation = 'relu', padding='same'),
            layers.MaxPooling2D(pool_size = (2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
            layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
            layers.MaxPooling2D(pool_size= (2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512, activation = 'relu'),
            layers.Dropout(0.5),
            layers.Dense(10,activation='softmax')
        ])
        return model

    model =  build_model()
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights = True)

    checkpoint = ModelCheckpoint("best_cifar10_model.h5",
                                 monitor='val_accuracy',
                                 save_best_only = True,
                                 mode='max',
                                 verbose = 1)

    X_train, X_val, y_train, y_val = train_test_split(
        train_images,
        train_labels,
        test_size=0.2,
        random_state=42
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=70,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=2
    )

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
    print(f"\nTest Doğruluğu (Accuracy): {test_acc:.4f}")

    y_pred_probs = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred_probs, axis = 1)
    y_true = test_labels.flatten()

    print("\nSınıflandırma Raporu:\n")
    print(classification_report(y_true, y_pred_classes, target_names = class_names))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize = (10,8))
    sns.heatmap(cm, annot=True, fmt = "d", cmap = "Blues",
                xticklabels = class_names,
                yticklabels=class_names)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix")
    plt.show()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.title('Doğruluk Grafiği')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label = 'Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Kayıp Grafiği')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
