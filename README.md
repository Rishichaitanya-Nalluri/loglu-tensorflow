## Installation

To use LogLU, install it using the following command:

```bash
pip install loglu-tensorflow
```

## Implementation

Import `LogLU` from the `loglu_tensorflow` package and use it directly as a Keras layer in your model.

```python
import tensorflow as tf
from loglu_tensorflow import LogLU

# Define a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128),
    LogLU(),  # Apply the LogLU activation function
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display the model architecture
model.summary()
```
## License

LogLU-TensorFlow is released under the [Apache License 2.0](https://github.com/Rishichaitanya-Nalluri/loglu-tensorflow/blob/main/LICENSE).
