# redneuronalMOK
Alondra Mishel Otero Mendoza, Brandon Osmar Pazos Trejo, Katya Maldonado Licona


import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import os
import tkinter as tk
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from scipy import ndimage

# Configuración global
TARGET_SIZE = (28, 28)
BATCH_SIZE = 64
EPOCHS = 100
N_SPLITS = 5

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

def advanced_preprocess_image(img, target_size=TARGET_SIZE):
    img = img.convert('L')
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = ndimage.gaussian_filter(img_array, sigma=0.5)
    return img_array.reshape(*target_size, 1)

def load_custom_images(base_dir='number_images', target_size=TARGET_SIZE):
    data, labels = [], []
    for i in range(1, 10):  # Ahora solo buscamos del 1 al 9
        folder_path = os.path.join(base_dir, str(i))
        if not os.path.exists(folder_path):
            print(f"Advertencia: La carpeta {folder_path} no existe.")
            continue
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, file_name)
                try:
                    img = Image.open(img_path)
                    img = advanced_preprocess_image(img, target_size)
                    data.append(img)
                    labels.append(i - 1)  # Restamos 1 para que los labels vayan de 0 a 8
                except Exception as e:
                    print(f"Error al procesar la imagen {img_path}: {str(e)}")
    return np.array(data), np.array(labels)

def create_advanced_model(input_shape=(28, 28, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Cargar y preparar los datos
print("Cargando imágenes personalizadas...")
X_custom, y_custom = load_custom_images()
print(f"Imágenes personalizadas cargadas: {len(X_custom)}")

print("Cargando MNIST...")
(X_mnist, y_mnist), (X_test, y_test) = load_mnist()
print(f"Imágenes MNIST cargadas: {len(X_mnist)}")

# Combinar datos personalizados con MNIST
X = np.concatenate([X_custom, X_mnist])
y = np.concatenate([y_custom, y_mnist])

# Mezclar los datos
X, y = shuffle(X, y, random_state=42)

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

# Preparar validación cruzada
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

trained_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"Entrenando pliegue {fold + 1}/{N_SPLITS}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = create_advanced_model()
    optimizer = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping]
    )
    
    trained_models.append(model)

def ensemble_predict(img):
    predictions = [model.predict(img) for model in trained_models]
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Dibuja un número en la cuadrícula")
        self.grid_size = 28
        self.cell_size = 20
        self.canvas = tk.Canvas(root, width=self.grid_size*self.cell_size, height=self.grid_size*self.cell_size, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = tk.Button(root, text="Predecir", command=self.predict_digit)
        self.predict_button.grid(row=1, column=1)
        
        self.clear_button = tk.Button(root, text="Borrar", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=2)

        self.label = tk.Label(root, text="Dibuja un número y luego haz clic en Predecir")
        self.label.grid(row=2, column=0, columnspan=4)

        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.draw_grid()

    def draw_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.canvas.create_rectangle(i*self.cell_size, j*self.cell_size, 
                                             (i+1)*self.cell_size, (j+1)*self.cell_size, 
                                             outline='light gray')

    def paint(self, event):
        x, y = event.x // self.cell_size, event.y // self.cell_size
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = 1
            self.canvas.create_rectangle(x*self.cell_size, y*self.cell_size, 
                                         (x+1)*self.cell_size, (y+1)*self.cell_size, 
                                         fill="black", outline='light gray')
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.draw_grid()
        self.label.config(text="Dibuja un número y luego haz clic en Predecir")

    def predict_digit(self):
        img = Image.fromarray((self.grid * 255).astype(np.uint8))
        img = advanced_preprocess_image(img)
        prediction = ensemble_predict(img.reshape(1, 28, 28, 1))
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        self.label.config(text=f"El número predicho es: {predicted_digit} (Confianza: {confidence:.2f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop() 
