import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths
base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG Splitted Dataset_Final/Reshaping Method'
test_dir = os.path.join(base_dir, 'test')
model_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG signals workflow/2D CNN models training,evaluation and results/LeNet-5 2D CNN results/Reshaping Method/training_run_1/final_model.h5'
output_dir = os.path.dirname(model_path)
output_file_path = os.path.join(output_dir, 'test_metrics.txt')
confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
metrics_plot_path = os.path.join(output_dir, 'metrics_plot.png')

# Class names (corresponding to folders in the test set)
class_names = ['Z', 'O', 'N', 'F', 'S']

# Initialize ImageDataGenerator for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the data generator for evaluation
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=8,
    class_mode='categorical',
    shuffle=False
)

# Load the model
model = tf.keras.models.load_model(model_path)

# Evaluate the model on test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Calculate metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

# Save overall metrics to the output file
with open(output_file_path, 'w') as f:
    f.write(f"Overall Evaluation Metrics:\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

# Generate classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print(report)
with open(output_file_path, 'a') as f:
    f.write("\nDetailed Classification Report:\n")
    f.write(report)

# Compute and save confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(confusion_matrix_path)
plt.close()

# Calculate and save TP, FP, FN, TN values for each class
TP, FP, FN, TN = {}, {}, {}, {}
for i, label in enumerate(class_names):
    TP[label] = cm[i, i]
    FP[label] = cm[:, i].sum() - TP[label]
    FN[label] = cm[i, :].sum() - TP[label]
    TN[label] = cm.sum() - (TP[label] + FP[label] + FN[label])

with open(output_file_path, 'a') as f:
    f.write("\nConfusion Matrix (with TP, FP, FN, TN values):\n")
    f.write(f"{cm}\n\n")
    for label in class_names:
        f.write(f"{label}:\n")
        f.write(f"  TP: {TP[label]}\n")
        f.write(f"  FP: {FP[label]}\n")
        f.write(f"  FN: {FN[label]}\n")
        f.write(f"  TN: {TN[label]}\n\n")

# Plot and save a bar plot of overall metrics
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('Overall Evaluation Metrics')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig(metrics_plot_path)
plt.show()
plt.close()

print("Evaluation complete. Metrics saved and confusion matrix plotted.")
