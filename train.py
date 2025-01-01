import argparse
import pandas as pd
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import  StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import tensorflow as tf


import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, GRU, LSTM, GlobalMaxPooling1D, Dropout

from sklearn.model_selection import train_test_split


#------------------ Let's build the function that will be used in the training------------------

def read_dataframe():
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    df = pd.read_csv("hf://datasets/christophsonntag/OLID/" + splits["train"])   
    return df

def clean_dataframe(df):
    selected_columns = ['tweet', 'subtask_a']
    df_simplified = df[selected_columns]
    df_simplified['target'] = 0
    df_simplified.loc[df_simplified['subtask_a'] == 'OFF', 'target'] = 1
    df_simplified = df_simplified.drop(columns=['subtask_a'])
    return df_simplified


def preprocess_dataframe(df):
    nlp = spacy.load("en_core_web_sm")
    nlp.Defaults.stop_words.add("user") 
    nlp.Defaults.stop_words.add("url")

    df["text_clean"] = df["tweet"].apply(lambda x:''.join(ch for ch in x if ch.isalnum() or ch==" "))
    df["text_clean"] = df["text_clean"].apply(lambda x: x.replace(" +"," ").lower().strip())
    df["text_clean"] = df["text_clean"].apply(lambda x: " ".join([token.lemma_ for token in nlp(x) if (token.lemma_ not in STOP_WORDS) & (token.text not in STOP_WORDS)]))

    return df 


def tokenize_dataframe(df, max_length=100):
    tokenizer = tf.keras.preprocessing.text.Tokenizer() # instanciate the tokenizer
    tokenizer.fit_on_texts(df["text_clean"])
    df["text_encoded"] = tokenizer.texts_to_sequences(df.text_clean)
    df["len_review"] = df["text_encoded"].apply(lambda x: len(x))
    df = df[df["len_review"]!=0]

    text_pad = tf.keras.preprocessing.sequence.pad_sequences(df.text_encoded, padding="post",maxlen=max_length)
    full_ds = tf.data.Dataset.from_tensor_slices((text_pad, df.target))

    return tokenizer, text_pad, full_ds

def build_model(tokenizer,input_shape_length=100,embedding_dim=32):
    vocab_size = len(tokenizer.word_index)
    model = tf.keras.Sequential([
                    Embedding(vocab_size+1, embedding_dim, input_shape=[input_shape_length,],name="embedding"),
                    GRU(units=64, return_sequences=True), # returns the last output
                    GlobalMaxPooling1D(),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(8, activation='relu'),
                    Dense(4, activation='relu'),
                    Dense(1, activation="sigmoid")
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Loss for binary classification
        metrics=['accuracy']
    )
    return model



if __name__ == "__main__":

    mlflow.tensorflow.autolog()

    with mlflow.start_run():

        # Time execution
        start_time = time.time()

        print("reading dataframe...")
        df = read_dataframe()
        print("dataframe read...")

        print("cleaning dataframe...")
        df = clean_dataframe(df)
        print("dataframe cleaned...")        

        print("preprocessing dataframe...")
        df = preprocess_dataframe(df)
        print("dataframe preprocessed...")  

        input_shape_length = 100

        print("tokenizing dataframe...")
        tokenizer, text_pad, full_ds = tokenize_dataframe(df,input_shape_length)
        print('tokenizing ok...')

        embedding_dim = 32
        print("building model...")
        model = build_model(tokenizer,input_shape_length,embedding_dim)

        xtrain, xval, ytrain, yval = train_test_split(text_pad,df.target, test_size=0.2)

        train = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        val = tf.data.Dataset.from_tensor_slices((xval, yval))
        train_batch = train.shuffle(len(train)).batch(64)
        val_batch = val.shuffle(len(val)).batch(64)


        print("training model...")
        model.fit(train_batch, epochs = 3, validation_data=val_batch)

        print("...Done!")
        print(f"---Total training time: {time.time()-start_time}")








    print("...Done!")