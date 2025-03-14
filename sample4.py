
#Transformer encoder for text classification
import random
import pandas
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
import string
import re
import numpy as np

vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 2048
sequence_length = 20
batch_size = 64

#preparing dataset
text_file = "kan.txt"
with open(text_file,encoding= "UTF-8") as f:
    lines = f.read().split("\n")[:-1]

text_pairs = []
for line in lines:
    english, kannada, ttmp = line.split("\t")
    kannada = "[start] " + kannada + " [end]"
    text_pairs.append((english, kannada))

num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]


#strip_chars = string.punctuation + "¿"
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace( lowercase, f"[{re.escape(strip_chars)}]", "")

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,output_mode="int",output_sequence_length=sequence_length,)

target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,output_mode="int",output_sequence_length=sequence_length +1,standardize=custom_standardization,)

train_english_texts = [pair[0] for pair in train_pairs]
train_kannada_texts = [pair[1] for pair in train_pairs]

source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_kannada_texts)

def format_dataset(eng, kann):
    eng = source_vectorization(eng)
    kann = target_vectorization(kann)
    return ({"english": eng,"kannada": kann[:, :-1],}, kann[:, 1:])


def make_dataset(pairs):
    eng_texts, kann_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    kann_texts = list(kann_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, kann_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()

#DataSet ready
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(2):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['kannada'].shape: {inputs['kannada'].shape}")
    print(f"targets.shape: {targets.shape}")
    # print(f"inputs['english']: {inputs['english']}")
    # print(f"inputs['kannada']: {inputs['kannada']}")
    # print(f"targets: {targets}")

# # actual code start
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding( input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding( input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    
    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim,"sequence_length": self.sequence_length,
                       "input_dim": self.input_dim,})
        return config

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"),layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim,"num_heads": self.num_heads,"dense_dim": self.dense_dim,})
        return config

class MyLayer(layers.Layer):
    def call(self, x):
        return PositionalEmbedding(sequence_length, vocab_size, embed_dim)(x)

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"),layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            causal_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(query=inputs,value=inputs,key=inputs,attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(query=attention_output_1,value=encoder_outputs,key=encoder_outputs,attention_mask=causal_mask,)
        attention_output_2 = self.layernorm_2(attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)
    
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1),tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim,"num_heads": self.num_heads,"dense_dim": self.dense_dim,})
        return config

# modal defanation starts
encoder_inputs = layers.Input(shape=(None,), dtype="int64", name="english")
x = MyLayer()(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = layers.Input(shape=(None,), dtype="int64", name="kannada")
x = MyLayer()(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)

transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
transformer.compile( optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [ keras.callbacks.ModelCheckpoint("transformer_en_decoder.keras", save_best_only=True)]
transformer.fit(train_ds, epochs=4, validation_data=train_ds,callbacks = callbacks)

print(f"Test acc: {transformer.evaluate(train_ds)[1]:.3f}")

kann_vocab = target_vectorization.get_vocabulary()
kann_index_lookup = dict(zip(range(len(kann_vocab)), kann_vocab))
max_decoded_sentence_length = 20

transformer.save_weights('TF_P_TRD.weights.h5')

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        print('words:'+str(i))
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = kann_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in train_pairs]
for _ in range(10):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
