import random
import pandas
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
import string
import re
import numpy as np


vocab_size = 15000
sequence_length = 20
batch_size = 64
embed_dim = 256
dense_dim = 2048
num_heads = 8

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

strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    return tf.strings.regex_replace( input_string, f"[{re.escape(strip_chars)}]", "")

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,output_mode="int",output_sequence_length=sequence_length,)

target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,output_mode="int",output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,)

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

for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['kannada'].shape: {inputs['kannada'].shape}")
    print(f"targets.shape: {targets.shape}")

test_eng_texts = [pair[0] for pair in val_pairs]
test_kann_texts = [pair[1] for pair in val_pairs]

for _ in range(10):
    input_sentence = random.choice(test_eng_texts)
    input_sentence1 = random.choice(test_kann_texts)
    print("-")
    print(input_sentence)
    print(input_sentence1)