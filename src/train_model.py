import tensorflow as tf
tf.config.list_physical_devices()
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data_preprocessing import load_and_preprocess_data
from src.model_architecture import create_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.visualize_performance import plot_performance


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

(x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

#x_train = x_train / 255.0
#x_test = x_test / 255.0

datagen.fit(x_train)

model = create_model()
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

history = model.fit(datagen.flow(x_train, y_train,batch_size=32), epochs=20, validation_data=(x_test,y_test), callbacks=[early_stopping, model_checkpoint])

plot_performance(history)
