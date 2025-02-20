from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

# Loading the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preparing the data for Random Forest
X_train_rf = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_rf = X_test.reshape(X_test.shape[0], -1) / 255.0

# Preparing the data for Neural Networks
X_train_nn = X_train_rf.copy()
X_test_nn = X_test_rf.copy()

# Preparing the data for CNN
X_train_cnn = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test_cnn = X_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encoding the labels for NN and CNN
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Interface for the MNIST Classifier
class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

# Random Forest Model with optimized hyperparameters
class RandomForestModel(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=12,  # Reduced the number of trees
            max_depth=5,      # Limit the maximum depth of the tree
            min_samples_split=5,  # Minimum number of samples for splitting
            n_jobs=-1         # Use all available CPU cores
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Feed-Forward Neural Network with optimized training time
class FeedForwardNNModel(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(784,)),  # Reduced the number of neurons
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_split=0.2)  # Reduced the number of epochs

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)

# Convolutional Neural Network with optimized training time
class CNNModel(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
            Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),  # Reduced the number of filters
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation='relu'),  # Reduced the number of neurons
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_split=0.2)  # Reduced the number of epochs

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)

# Classifier Manager
class MnistClassifier:
    def __init__(self, algorithm: str):
        if algorithm == 'rf':
            self.model = RandomForestModel()
        elif algorithm == 'nn':
            self.model = FeedForwardNNModel()
        elif algorithm == 'cnn':
            self.model = CNNModel()
        else:
            raise ValueError("Algorithm not recognized")

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Main loop for user input
while True:
    print("\nAvailable models:")
    print("1. Random Forest (rf)")
    print("2. Feed-Forward Neural Network (nn)")
    print("3. Convolutional Neural Network (cnn)")
    print("Type 'end' to exit.")
    
    algorithm_choice = input("Choose a model to run (rf, nn, cnn): ").strip().lower()

    if algorithm_choice == 'end':
        print("Exiting the program...")
        break
    elif algorithm_choice not in ['rf', 'nn', 'cnn']:
        print("Invalid choice. Please choose one of the available models.")
        continue

    models = {
        'rf': (X_train_rf, y_train, X_test_rf),
        'nn': (X_train_nn, y_train_cat, X_test_nn),
        'cnn': (X_train_cnn, y_train_cat, X_test_cnn)
    }

    # Output dataset shape for Random Forest
    print("Dataset shape:", models[algorithm_choice][0].shape, models[algorithm_choice][2].shape)

    # Running the selected model
    print(f"\nTraining and predicting using {algorithm_choice.upper()} model...")
    classifier = MnistClassifier(algorithm=algorithm_choice)
    classifier.train(models[algorithm_choice][0], models[algorithm_choice][1])
    predictions = classifier.predict(models[algorithm_choice][2])
    print(f"Sample predictions ({algorithm_choice.upper()}):", predictions[:10])

    # Checking the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy ({algorithm_choice.upper()}): {accuracy:.4f}")