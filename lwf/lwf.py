import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data (normalize and reshape)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Split the dataset into two tasks (0-4) and (5-9)
task_1_mask = np.isin(y_train, [0, 1, 2, 3, 4])
task_2_mask = np.isin(y_train, [5, 6, 7, 8, 9])

x_train_task_1 = x_train[task_1_mask]
y_train_task_1 = y_train[task_1_mask]
x_train_task_2 = x_train[task_2_mask]
y_train_task_2 = y_train[task_2_mask]

# Build the model
def build_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train on task 1 (Digits 0-4)
model_task_1 = build_model()
model_task_1.fit(x_train_task_1, y_train_task_1, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Save the model to retain performance on task 1
model_task_1.save("task_1_model.h5")

# Now we will load the model to continue training without forgetting
model_task_1 = tf.keras.models.load_model("task_1_model.h5")

# Generate the soft targets for task 2 (Digits 5-9)
# We will use the model trained on task 1 to generate soft targets for task 2
y_pred_task_1 = model_task_1.predict(x_train_task_2)

# Train on task 2 (Digits 5-9) with LwF
# Use the outputs from task 1 as soft targets, alongside the true labels of task 2
# The labels must be integers, not one-hot encoded for sparse categorical cross entropy
def lwf_loss(y_true, y_pred):
    # True loss (standard sparse categorical cross-entropy)
    true_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Soft loss (KL-divergence between the predictions of the new task and the old model's predictions)
    batch_size = tf.shape(y_pred)[0]  # Get the current batch size
    y_pred_task_1_resized = tf.slice(y_pred_task_1, [0, 0], [batch_size, -1])  # Slice to match batch size
    
    soft_loss = tf.keras.losses.KLDivergence()(y_pred, y_pred_task_1_resized)
    
    return true_loss + soft_loss

# Rebuild model with LwF loss function
model_task_2 = build_model()
model_task_2.compile(optimizer='adam', loss=lwf_loss, metrics=['accuracy'])

# Train on task 2 using both soft targets and ground truth labels
model_task_2.fit(x_train_task_2, y_train_task_2, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate performance on both tasks
print("Performance on Task 1 (Digits 0-4):")
task_1_accuracy = model_task_2.evaluate(x_train_task_1, y_train_task_1)
print(f"Task 1 Accuracy: {task_1_accuracy[1] * 100:.2f}%")

print("\nPerformance on Task 2 (Digits 5-9):")
task_2_accuracy = model_task_2.evaluate(x_train_task_2, y_train_task_2)
print(f"Task 2 Accuracy: {task_2_accuracy[1] * 100:.2f}%")
