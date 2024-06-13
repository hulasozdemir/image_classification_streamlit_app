import tensorflow as tf
tf.config.list_physical_devices()
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data_preprocessing import load_and_preprocess_data
from src.model_architecture import create_model

(x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

model = create_model()
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

history = model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
