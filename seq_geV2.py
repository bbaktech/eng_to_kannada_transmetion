from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras import layers
import numpy as np

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
    
def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)

class MyLayer(layers.Layer):
    def call(self, x):
        return PositionalEmbedding(sequence_length, vocab_size, embed_dim)(x)

dataset = keras.utils.text_dataset_from_directory(directory="aclImdb", label_mode=None, batch_size=32)
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))

sequence_length = 25
vocab_size = 20
text_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,output_mode="int",ngrams=2,output_sequence_length=sequence_length,)
#    max_tokens=vocab_size,output_mode="int")

text_vectorization.adapt(dataset)

def prepare_lm_dataset(text_batch):
    vectorized_sequences = text_vectorization(text_batch)
    #omots last column
    x = vectorized_sequences[:, :-1]
    #omits first column
    y = vectorized_sequences[:, 1:]
    return x, y

lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)

embed_dim = 256
latent_dim = 256
num_heads = 2

inputs = keras.Input(shape=(None,), dtype="int64")
x = MyLayer()(inputs) #positional embeding layer
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop",metrics=['accuracy'])

tokens_index = dict(enumerate(text_vectorization.get_vocabulary()))
#list of words printed
print (tokens_index)

def sample_next(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

# class TextGenerator(keras.callbacks.Callback):
#     def __init__(self,
#             prompt,
#             generate_length,
#             model_input_length,
#             temperatures=(1.,),
#             print_freq=1):
#         self.prompt = prompt
#         self.generate_length = generate_length
#         self.model_input_length = model_input_length
#         self.temperatures = temperatures
#         self.print_freq = print_freq

#     def on_epoch_end(self, epoch, logs=None):
#         if (epoch + 1) % self.print_freq != 0:
#             return
#         for temperature in self.temperatures:
#             print("== Generating with temperature", temperature)
#             sentence = self.prompt
#             for i in range(self.generate_length):
#                 tokenized_sentence = text_vectorization([sentence])
#                 predictions = self.model(tokenized_sentence)
#                 next_token = sample_next(predictions[0, i, :])
#                 print('next token:'+ str(next_token))
#                 sampled_token = tokens_index[next_token]
#                 sentence += " " + sampled_token
#             print(sentence)
# prompt = "aa"
# text_gen_callback = TextGenerator(
#                         prompt,
#                         generate_length=10,
#                         model_input_length=sequence_length,
#                         temperatures=(0.2, 0.5, 0.7, 1., 1.5))

#model.fit(lm_dataset, epochs=100, callbacks=[text_gen_callback])

history = model.fit(lm_dataset, epochs=100)

# Plotting the training history
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

model.save_weights('TF_SEQ_SEQV2.weights.h5')

model.load_weights('TF_SEQ_SEQV2.weights.h5')
def textGenaration(prompt,
                        generate_length=10,
                        model_input_length=sequence_length,
                        temperatures=(0.2, 0.5, 0.7, 1., 1.5)):
    for temperature in temperatures:
        print("== Generating with temperature", temperature)
        sentence = prompt
        for i in range(generate_length):
            tokenized_sentence = text_vectorization([sentence])
            predictions = model(tokenized_sentence)
            next_token = sample_next(predictions[0, i, :])
#            print('next token:'+ str(next_token))
            sampled_token = tokens_index[next_token]
            sentence += " " + sampled_token
        print(sentence)

prompt = "44 55 66 77 11 22"
textGenaration(prompt)
