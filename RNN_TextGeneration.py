#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import glob

# 1. Load and read the text dataset
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)
text = open(path_to_file, "rb").read().decode("utf-8")
print(f"Corpus length: {len(text)} characters")

# 2. Create character ↔ index mappings
vocab = sorted(set(text))
vocab_size = len(vocab)
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)

# 3. Vectorize the text
text_as_int = np.array([char2idx[c] for c in text])

# 4. Split into input-target sequence pairs
seq_length = 100
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)

# 5. Batch, shuffle, and prefetch
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# 6. Build the stateful LSTM model (Functional API)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense

embedding_dim = 256
rnn_units = 1024

train_inputs = Input(batch_shape=(BATCH_SIZE, None), dtype=tf.int32)
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(train_inputs)
x = LSTM(
    rnn_units,
    return_sequences=True,
    stateful=True,
    recurrent_initializer="glorot_uniform"
)(x)
train_outputs = Dense(vocab_size)(x)

model = Model(train_inputs, train_outputs)

# 7. Compile with sparse categorical crossentropy
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer="adam", loss=loss)

# 8. Configure checkpoints (.weights.h5 required for save_weights_only)
checkpoint_dir = "./training_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5"),
    save_weights_only=True,
    verbose=1
)

# 9. Train the model
EPOCHS = 10
model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# 10. Text-generation helper (uses temperature scaling)
def sample(start_string: str, num_generate: int = 500, temperature: float = 1.0):
    # Rebuild the model with batch_size = 1 for generation
    gen_inputs = Input(batch_shape=(1, None), dtype=tf.int32)
    y = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(gen_inputs)
    y = LSTM(
        rnn_units,
        return_sequences=True,
        stateful=True,
        recurrent_initializer="glorot_uniform",
        name="gen_lstm"
    )(y)
    gen_outputs = Dense(vocab_size)(y)
    gen_model = Model(gen_inputs, gen_outputs)

    # Find the latest .weights.h5 checkpoint
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "ckpt_*.weights.h5"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    latest = max(ckpt_files, key=os.path.getmtime)

    # Load weights and build
    gen_model.load_weights(latest)
    gen_model.build(tf.TensorShape([1, None]))

    # Reset the LSTM layer’s states
    gen_model.get_layer("gen_lstm").reset_states()

    # Prepare the input
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Generate characters
    text_generated = []
    for _ in range(num_generate):
        preds = gen_model(input_eval)
        preds = tf.squeeze(preds, 0) / temperature
        predicted_id = tf.random.categorical(preds, num_samples=1)[-1, 0].numpy()

        text_generated.append(idx2char[predicted_id])
        input_eval = tf.expand_dims([predicted_id], 0)

    return start_string + "".join(text_generated)

if __name__ == "__main__":
    # Example: lower temperature => more deterministic output
    print(sample(start_string="ROMEO: ", temperature=0.5))
