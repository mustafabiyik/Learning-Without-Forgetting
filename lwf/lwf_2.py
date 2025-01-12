import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data (normalize and reshape)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Split the training data into a training and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Build a basic neural network model for binary classification
def build_model(num_classes=1):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
        layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')  # Softmax for multi-class
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy', metrics=['accuracy'])
    return model

# Define the Learning without Forgetting (LwF) loss function
def lwf_loss(y_true, y_pred, model_task_old=None, old_labels=None):
    # Cross-entropy loss for current task
    true_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Soft loss for previous tasks (if model_task_old is provided)
    if model_task_old is not None:
        y_pred_old = model_task_old.predict(x_train)  # Soft targets from previous model
        soft_loss = tf.reduce_mean(tf.keras.losses.KLDivergence()(y_pred, y_pred_old))  # KL Divergence for soft target matching
        return true_loss + soft_loss
    return true_loss

# Task Training Function with LwF and tracking history for plotting
def train_task(task_num, model_task, model_task_old=None, previous_labels=None):
    y_train_task = np.where(np.isin(y_train, previous_labels), y_train, task_num)  # Task-specific labels
    y_val_task = np.where(np.isin(y_val, previous_labels), y_val, task_num)  # Task-specific labels
    
    history = model_task.fit(x_train, y_train_task, epochs=5, batch_size=32, validation_data=(x_val, y_val_task))
    
    return history

# Step 1: Train Task 1 (Classifying digit "0" vs others)
model_task_1 = build_model(num_classes=2)  # "0" vs others
y_train_task_1 = np.where(y_train == 0, 1, 0)
y_val_task_1 = np.where(y_val == 0, 1, 0)
history_task_1 = model_task_1.fit(x_train, y_train_task_1, epochs=5, batch_size=32, validation_data=(x_val, y_val_task_1))
model_task_1.save("task_1_model.h5")

# Step 2: Train Task 2 (Classifying "0", "1" vs others)
model_task_2 = build_model(num_classes=3)  # "0", "1" vs others
model_task_2.load_weights("task_1_model.h5")
history_task_2 = train_task(task_num=2, model_task=model_task_2, model_task_old=model_task_1, previous_labels=[0])

# Step 3: Train Task 3 (Classifying "0", "1", "2" vs others)
model_task_3 = build_model(num_classes=4)  # "0", "1", "2" vs others
model_task_3.load_weights("task_2_model.h5")
history_task_3 = train_task(task_num=3, model_task=model_task_3, model_task_old=model_task_2, previous_labels=[0, 1])

# Step 4: Train Task 4 (Classifying "0", "1", "2", "3" vs others)
model_task_4 = build_model(num_classes=5)  # "0", "1", "2", "3" vs others
model_task_4.load_weights("task_3_model.h5")
history_task_4 = train_task(task_num=4, model_task=model_task_4, model_task_old=model_task_3, previous_labels=[0, 1, 2])

# Continue similarly for all tasks till Task 10 (All digits)

# Task 10 (Classifying "0" to "9" vs others)
model_task_10 = build_model(num_classes=10)  # "0" to "9" vs others
model_task_10.load_weights("task_9_model.h5")
history_task_10 = train_task(task_num=10, model_task=model_task_10, model_task_old=model_task_9, previous_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

# Plot training and validation accuracy/loss for each task
def plot_history(history, task_num):
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Task {task_num} - Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Task {task_num} - Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# Plot the accuracy and loss for Task 1 to Task 10
plot_history(history_task_1, 1)
plot_history(history_task_2, 2)
plot_history(history_task_3, 3)
plot_history(history_task_4, 4)
plot_history(history_task_10, 10)
