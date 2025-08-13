import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same', strides=stride)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet_cifar10():
    inputs = layers.Input(shape=(32,32,3))
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

def main():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog','frog','horse','ship','truck']

    mean = np.mean(train_images, axis=(0,1,2), keepdims=True)
    std = np.std(train_images, axis=(0,1,2), keepdims=True)
    train_images = (train_images - mean)/std
    test_images = (test_images - mean)/std

    train_labels_cat = to_categorical(train_labels, 10)
    test_labels_cat = to_categorical(test_labels, 10)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8,1.2]
    )
    datagen.fit(train_images)

    model = build_resnet_cifar10()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    optimizer = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint("best_resnet_cifar10.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels_cat, test_size=0.2, random_state=42)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        epochs=80,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint, lr_scheduler],
        verbose=2
    )

    test_loss, test_acc = model.evaluate(test_images, test_labels_cat, verbose=2)
    print(f"\nTest Doğruluğu (Accuracy): {test_acc:.4f}")

    y_pred_probs = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(test_labels_cat, axis=1)

    print("\nSınıflandırma Raporu:\n")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix")
    plt.show()

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.title('Doğruluk Grafiği')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Kayıp Grafiği')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

