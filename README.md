
### Installation

To use LogLU, simply install it using the following command:

```bash
pip install loglu-tensorflow
```

### Implementation

```bash
import tensorflow as tf
from loglu_tensorflow import LogLU

# Define a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128),
    LogLU(),  # Apply LogLU directly as a Keras layer
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Summary of the model
model.summary()
```

## License

LogLU-TensorFlow is released under the [Apache License 2.0](https://github.com/Rishichaitanya-Nalluri/loglu-tensorflow/blob/main/LICENSE).
