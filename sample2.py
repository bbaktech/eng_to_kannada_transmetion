
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
dense_dim = 32

df1 = pandas.read_csv('human_chatgpt_genarated_dataset.csv')
X = df1.iloc[:, 0]
Y  = df1.iloc[:, 1]
print (len(X))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)

text_vectorization = layers.TextVectorization(max_tokens=vocab_size,output_mode="int")
text_vectorization.adapt(X)

print ( 'vocabulary size:' + str(len(text_vectorization.get_vocabulary())))

encode_X_train = text_vectorization(X_train)
encode_X_test = text_vectorization(X_test)

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

inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
callbacks = [ keras.callbacks.ModelCheckpoint("transformer_enkeras", save_best_only=True)]
#callbacks = [ keras.callbacks.ModelCheckpoint("transformer_encoder.keras", save_best_only=True)]
model.fit( encode_X_train, Y_train, epochs=5, callbacks=callbacks)

model.save_weights('TF_EN.weights.h5')
model.load_weights('TF_EN.weights.h5')
print(f"Test acc: {model.evaluate(encode_X_test,Y_test)[1]:.3f}")

print("predictions: New values")
xx_test = [' مجموع 5 و 7 و 9 يساوي 21.']
#should get 1
encode_xx_test = text_vectorization(xx_test)
pr = model.predict(encode_xx_test)
print(xx_test)
for prd in pr:
    if prd[0] > 0.5 :
        p = 1
    else :
        p = 0
    print('p:' + str(p) + ' val:' + str(prd[0]))

print("predictions: old values")
predictions = model.predict(encode_X_test)

i =-1
for y in Y_test :
    i+=1
    if predictions[i][0] > 0.5 :
        p = 1
    else :
        p = 0
    print('predection:' + str(p) + ' val:' + str(predictions[i][0]) +' actual y:'+str( y ))
