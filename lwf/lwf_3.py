
# LwF (Learning without Forge ng) wit
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

def evaluate_task(model, x_test, y_test_task, num_classes):
    #restrict test labels to the valid range
    y_test_task = np.clip(y_test_task, 0, num_classes - 1)
    #evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test_task, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy

#function to handle training and testing with evaluation after each task
def train_and_evaluate_task(task_num, model_task, model_task_old=None, previous_labels=None, x_test=x_test, y_test=y_test, num_classes=3):
    # Train the task
    y_train_task = np.where(np.isin(y_train, previous_labels), y_train, task_num)  # Task-specific labels
    y_val_task = np.where(np.isin(y_val, previous_labels), y_val, task_num)  # Task-specific labels

    history = model_task.fit(x_train, y_train_task, epochs=5, batch_size=32, validation_data=(x_val, y_val_task))
    
    # Evaluate the model on the test set
    y_test_task = np.where(np.isin(y_test, previous_labels), y_test, task_num)  # Task-specific test labels
    test_loss, test_accuracy = evaluate_task(model_task, x_test, y_test_task, num_classes)
    
    return history, test_loss, test_accuracy

# Step 1: Train Task 1 -- digit "0" vs others
model_task_1 = build_model(num_classes=2)  # "0" vs others
y_train_task_1 = np.where(y_train == 0, 1, 0)
y_val_task_1 = np.where(y_val == 0, 1, 0)
history_task_1 = model_task_1.fit(x_train, y_train_task_1, epochs=5, batch_size=32, validation_data=(x_val, y_val_task_1))
test_loss_1, test_accuracy_1 = evaluate_task(model_task_1, x_test, y_test, num_classes=2)
model_task_1.save("task_1_model.h5")

# Step 2: Train Task 2 -- 0, 1 vs others
model_task_2 = build_model(num_classes=3)  # 0, 1 vs others
#load weights from the previous model for the common layers
#copy shared layers
model_task_2.layers[0].set_weights(model_task_1.layers[0].get_weights())
history_task_2, test_loss_2, test_accuracy_2 = train_and_evaluate_task(task_num=2, model_task=model_task_2, model_task_old=model_task_1, previous_labels=[0], num_classes=3)

# Step 3: Train Task 3 -- 0, 1, 2 vs others
model_task_3 = build_model(num_classes=4)  # 0, 1, 2 vs others
#load weights from the previous model for the common layers
#copy shared layers
model_task_3.layers[0].set_weights(model_task_2.layers[0].get_weights())  
history_task_3, test_loss_3, test_accuracy_3 = train_and_evaluate_task(task_num=3, model_task=model_task_3, model_task_old=model_task_2, previous_labels=[0, 1], num_classes=4)

# Step 4: Train Task 4 -- 0, 1, 2, 3  vs others
model_task_4 = build_model(num_classes=5) 
#load weights from the previous model for the common layers
#copy shared layers
model_task_4.layers[0].set_weights(model_task_3.layers[0].get_weights())  
history_task_4, test_loss_4, test_accuracy_4 = train_and_evaluate_task(task_num=4, model_task=model_task_4, model_task_old=model_task_3, previous_labels=[0, 1, 2], num_classes=5)

# Step 5: Train Task 5 
model_task_5 = build_model(num_classes=6) 
#load weights from the previous model for the common layers
#copy shared layers
model_task_5.layers[0].set_weights(model_task_4.layers[0].get_weights())  
history_task_5, test_loss_5, test_accuracy_5 = train_and_evaluate_task(task_num=5, model_task=model_task_5, model_task_old=model_task_4, previous_labels=[0, 1, 2, 3], num_classes=6)

# Step 6: Train Task 6 -- 0, 1, 2, 3, 4, 5  vs others
model_task_6 = build_model(num_classes=7) 
#load weights from the previous model for the common layers
#copy shared layers
model_task_6.layers[0].set_weights(model_task_5.layers[0].get_weights())  
history_task_6, test_loss_6, test_accuracy_6 = train_and_evaluate_task(task_num=6, model_task=model_task_6, model_task_old=model_task_5, previous_labels=[0, 1, 2, 3, 4], num_classes=7)

# Step 7: Train Task 7
model_task_7 = build_model(num_classes=8) 
#load weights from the previous model for the common layers
#copy shared layers
model_task_7.layers[0].set_weights(model_task_6.layers[0].get_weights())  
history_task_7, test_loss_7, test_accuracy_7 = train_and_evaluate_task(task_num=7, model_task=model_task_7, model_task_old=model_task_6, previous_labels=[0, 1, 2, 3, 4, 5], num_classes=8)

# Step 8: Train Task 8 
model_task_8 = build_model(num_classes=9) 
#load weights from the previous model for the common layers
#copy shared layers
model_task_8.layers[0].set_weights(model_task_7.layers[0].get_weights())  
history_task_8, test_loss_8, test_accuracy_8 = train_and_evaluate_task(task_num=8, model_task=model_task_8, model_task_old=model_task_7, previous_labels=[0, 1, 2, 3, 4, 5, 6], num_classes=9)

# Step 9: Train Task 9 
model_task_9 = build_model(num_classes=10) 
#load weights from the previous model for the common layers
#copy shared layers
model_task_9.layers[0].set_weights(model_task_8.layers[0].get_weights())  
history_task_9, test_loss_9, test_accuracy_9 = train_and_evaluate_task(task_num=9, model_task=model_task_9, model_task_old=model_task_8, previous_labels=[0, 1, 2, 3, 4, 5, 6, 7], num_classes=10)


# Step 10: Train Task 10 
#model_task_10 = build_model(num_classes=10) 
##load weights from the previous model for the common layers
##copy shared layers
#model_task_10.layers[0].set_weights(model_task_9.layers[0].get_weights())  
#history_task_10 = train_task(task_num=10, model_task=model_task_10, model_task_old=model_task_9, previous_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])


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
    plt.savefig(f"task-{task_num}_accuracy_loss_plot.png") 
    plt.show()

# Plot the accuracy and loss for Task 1 to Task 10
plot_history(history_task_1, 1)
plot_history(history_task_2, 2)
plot_history(history_task_3, 3)
plot_history(history_task_4, 4)
plot_history(history_task_5, 5)
plot_history(history_task_6, 6)
plot_history(history_task_7, 7)
plot_history(history_task_8, 8)
plot_history(history_task_9, 9)

print(f"Test Accuracy for Task 1: {test_accuracy_1:.4f}, Test Loss: {test_loss_1:.4f}")
print(f"Test Accuracy for Task 2: {test_accuracy_2:.4f}, Test Loss: {test_loss_2:.4f}")
print(f"Test Accuracy for Task 3: {test_accuracy_3:.4f}, Test Loss: {test_loss_3:.4f}")
print(f"Test Accuracy for Task 4: {test_accuracy_4:.4f}, Test Loss: {test_loss_4:.4f}")
print(f"Test Accuracy for Task 5: {test_accuracy_5:.4f}, Test Loss: {test_loss_5:.4f}")
print(f"Test Accuracy for Task 6: {test_accuracy_6:.4f}, Test Loss: {test_loss_6:.4f}")
print(f"Test Accuracy for Task 7: {test_accuracy_7:.4f}, Test Loss: {test_loss_7:.4f}")
print(f"Test Accuracy for Task 8: {test_accuracy_8:.4f}, Test Loss: {test_loss_8:.4f}")
print(f"Test Accuracy for Task 9: {test_accuracy_9:.4f}, Test Loss: {test_loss_9:.4f}")
#print(f"Test Accuracy for Task 10: {test_accuracy_10:.4f}, Test Loss: {test_loss_10:.4f}")
#plot_history(history_task_10, 10)

"""
Test Accuracy for Task 1: 1.0000, Test Loss: 0.0003
Test Accuracy for Task 2: 0.9981, Test Loss: 0.0085
Test Accuracy for Task 3: 0.9956, Test Loss: 0.0196
Test Accuracy for Task 3: 0.9956, Test Loss: 0.0196
Test Accuracy for Task 3: 0.9956, Test Loss: 0.0196
Test Accuracy for Task 3: 0.9956, Test Loss: 0.0196
Test Accuracy for Task 3: 0.9956, Test Loss: 0.0196
Test Accuracy for Task 4: 0.9921, Test Loss: 0.0316
Test Accuracy for Task 5: 0.9855, Test Loss: 0.0478
Test Accuracy for Task 6: 0.9818, Test Loss: 0.0665
Test Accuracy for Task 7: 0.9799, Test Loss: 0.0691
Test Accuracy for Task 8: 0.9802, Test Loss: 0.0751
Test Accuracy for Task 9: 0.9780, Test Loss: 0.0838

"""