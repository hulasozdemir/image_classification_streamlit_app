from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_preprocessing import load_and_preprocess_data
import numpy as np

(x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
model = load_model('best_model.keras')

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

y_pred = model.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print(classification_report(y_test, y_pred_classes, target_names=class_names))

