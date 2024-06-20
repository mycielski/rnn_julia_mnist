from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, InputLayer, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Hyperparameters:
TIME_STEPS = 4
INPUT_SIZE = int(28 * 28 / 4)
OUTPUT_SIZE = 10
HIDDEN_SIZE = 64
BATCH_SIZE = 100
EPOCHS = 5
LEARNING_RATE = 15e-3

# Load data:
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
x_train_processed = x_train.reshape(-1, TIME_STEPS, INPUT_SIZE).astype("float32") / 255
x_test_processed = x_test.reshape(-1, TIME_STEPS, INPUT_SIZE).astype("float32") / 255
y_train_categorical = to_categorical(y_train, num_classes=OUTPUT_SIZE)
y_test_categorical = to_categorical(y_test, num_classes=OUTPUT_SIZE)

# Build model:
model = Sequential()
model.add(InputLayer(shape=(TIME_STEPS, INPUT_SIZE)))
model.add(SimpleRNN(HIDDEN_SIZE, activation="tanh", return_sequences=False))
model.add(Dense(OUTPUT_SIZE, activation="softmax"))
model.compile(
    optimizer=Adam(LEARNING_RATE), loss="categorical_crossentropy", metrics=["accuracy"]
)

# Train model:
model.fit(x_train_processed, y_train_categorical, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Test model:
loss, accuracy = model.evaluate(x_test_processed, y_test_categorical)
print(f"Test accuracy: {100*accuracy:.2f}%")
