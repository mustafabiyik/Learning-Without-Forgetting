import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Define subsets for tasks
def get_task_data(x, y, task_labels):
    mask = np.isin(y, task_labels)
    x_task = x[mask]
    y_task = y[mask]
    y_task = np.array([task_labels.index(label) for label in y_task])
    return x_task, to_categorical(y_task, len(task_labels))

# Model definition
def create_model(input_dim, output_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# EWC regularization
class EWC:
    def __init__(self, model, dataset, fisher_samples=200):
        self.model = model
        self.dataset = dataset
        self.fisher_samples = fisher_samples
        self.fisher_matrix = None
        self.optimal_weights = None

    def compute_fisher_information(self):
        gradients = []
        for _ in range(self.fisher_samples):
            idx = np.random.choice(len(self.dataset[0]))
            x_sample = self.dataset[0][idx:idx + 1]
            y_sample = self.dataset[1][idx:idx + 1]
            with tf.GradientTape() as tape:
                preds = self.model(x_sample)
                loss = tf.keras.losses.categorical_crossentropy(y_sample, preds)
            grads = tape.gradient(loss, self.model.trainable_variables)
            gradients.append([g.numpy() for g in grads])

        fisher_matrix = [np.mean([g[i] ** 2 for g in gradients], axis=0) for i in range(len(gradients[0]))]
        self.fisher_matrix = fisher_matrix
        self.optimal_weights = [w.numpy() for w in self.model.trainable_variables]

    def ewc_loss(self, new_loss):
        if self.fisher_matrix is None:
            return new_loss

        ewc_penalty = 0
        for i, var in enumerate(self.model.trainable_variables):
            penalty = self.fisher_matrix[i] * (var - self.optimal_weights[i]) ** 2
            ewc_penalty += tf.reduce_sum(penalty)

        return new_loss + 0.5 * ewc_penalty

# Training pipeline
def train_task(model, x_train_task, y_train_task, ewc=None, epochs=5):
    history = {"loss": [], "accuracy": []}
    if ewc is not None:
        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                preds = model(x_batch)
                loss = tf.keras.losses.categorical_crossentropy(y_batch, preds)
                if ewc is not None:
                    loss = ewc.ewc_loss(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        for epoch in range(epochs):
            epoch_loss = []
            correct_predictions = 0
            total_samples = 0
            for i in range(0, len(x_train_task), 32):
                x_batch = x_train_task[i:i + 32]
                y_batch = y_train_task[i:i + 32]
                loss = train_step(x_batch, y_batch)
                epoch_loss.append(loss.numpy().mean())

                preds = model.predict(x_batch, verbose=0)
                correct_predictions += np.sum(np.argmax(preds, axis=1) == np.argmax(y_batch, axis=1))
                total_samples += len(y_batch)

            history["loss"].append(np.mean(epoch_loss))
            history["accuracy"].append(correct_predictions / total_samples)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {history['loss'][-1]:.4f}, Accuracy: {history['accuracy'][-1]:.4f}")
    else:
        hist = model.fit(x_train_task, y_train_task, epochs=epochs, batch_size=32, verbose=2)
        history["loss"] = hist.history["loss"]
        history["accuracy"] = hist.history["accuracy"]

    return history

# Plot accuracy and loss
def plot_metrics(history, task_name):
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Loss")
    plt.title(f"{task_name} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], label="Accuracy")
    plt.title(f"{task_name} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

# Task 1: Train on "1 vs others"
task_1_labels = [1, -1]
x_train_1, y_train_1 = get_task_data(x_train, y_train, task_1_labels)
x_test_1, y_test_1 = get_task_data(x_test, y_test, task_1_labels)

model = create_model(28 * 28, len(task_1_labels))
history_task_1 = train_task(model, x_train_1, y_train_1)
plot_metrics(history_task_1, "Task 1")
print("Task 1 Accuracy:", model.evaluate(x_test_1, y_test_1, verbose=0))

# Save EWC information
ewc = EWC(model, (x_train_1, y_train_1))
ewc.compute_fisher_information()

# Task 2: Train on "1, 2, others" without forgetting Task 1
task_2_labels = [1, 2, -1]
x_train_2, y_train_2 = get_task_data(x_train, y_train, task_2_labels)
x_test_2, y_test_2 = get_task_data(x_test, y_test, task_2_labels)

# Modify model for new task
model = create_model(28 * 28, len(task_2_labels))
history_task_2 = train_task(model, x_train_2, y_train_2, ewc=ewc)
plot_metrics(history_task_2, "Task 2")
print("Task 2 Accuracy:", model.evaluate(x_test_2, y_test_2, verbose=0))

# Save EWC information after Task 2
ewc = EWC(model, (x_train_2, y_train_2))
ewc.compute_fisher_information()

# Task 3: Train on "1, 2, 3, others" without forgetting previous tasks
task_3_labels = [1, 2, 3, -1]
x_train_3, y_train_3 = get_task_data(x_train, y_train, task_3_labels)
x_test_3, y_test_3 = get_task_data(x_test, y_test, task_3_labels)

# Modify model for new task
model = create_model(28 * 28, len(task_3_labels))
history_task_3 = train_task(model, x_train_3, y_train_3, ewc=ewc)
plot_metrics(history_task_3, "Task 3")
print("Task 3 Accuracy:", model.evaluate(x_test_3, y_test_3, verbose=0))
