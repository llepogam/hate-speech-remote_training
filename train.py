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

from tensorflow.keras.layers import TextVectorization


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


def build_model(input_shape_length=100,embedding_dim=32,vocab_size=20000):
        
    vectorizer = TextVectorization(
        max_tokens=vocab_size, 
        output_mode='int',
        output_sequence_length=input_shape_length
    )
    
    model = tf.keras.Sequential([
                    vectorizer,
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
    return model, vectorizer


#------------------ Let's here write the flow of the MLFlow run------------------

if __name__ == "__main__":

    # Set MLflow experiment name
    experiment_name = "hate_speech_detection"
    mlflow.set_experiment(experiment_name)

    mlflow.tensorflow.autolog()  # Enable automatic logging for TensorFlow


    with mlflow.start_run():

        # Time execution
        start_time = time.time()

        print("reading dataframe...")
        df = read_dataframe()
        print("dataframe read...")

        print("cleaning dataframe...")
        df_clean = clean_dataframe(df)
        print("dataframe cleaned...")        

        print("preprocessing dataframe...")
        df_preprocessed = preprocess_dataframe(df_clean)
        print("dataframe preprocessed...")  

        input_shape_length = 100

        print("splitting the data...")
        xtrain, xval, ytrain, yval = train_test_split(df_preprocessed["text_clean"], df_preprocessed.target, test_size=0.2)

        train = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        val = tf.data.Dataset.from_tensor_slices((xval, yval))
        train_batch = train.shuffle(len(train)).batch(64)
        val_batch = val.shuffle(len(val)).batch(64)
        print("data split..")        

        embedding_dim = 32
        print("building model...")
        model,vectorizer = build_model(input_shape_length,embedding_dim)

        # Adapt the vectorizer to the training data
        print("adapting vectorizer...")
        vectorizer.adapt(train.map(lambda x, y: x))

        # Log parameters
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("input_shape_length", input_shape_length)
        mlflow.log_param("batch_size", 64)


        print("training model...")
        history = model.fit(train_batch, epochs = 3, validation_data=val_batch)

        # Log metrics for each epoch
        for epoch in range(len(history.history['loss'])):
            for metric_name in history.history.keys():
                mlflow.log_metric(metric_name, history.history[metric_name][epoch], step=epoch)

        # Infer and log the model signature
        input_sample = next(iter(train_batch.take(1)))[0]  # Get a sample batch of input
        predicted_sample = model.predict(input_sample)
        signature = infer_signature(input_sample.numpy(), predicted_sample)
        mlflow.tensorflow.log_model(model, artifact_path="model", signature=signature)



        print("...Done!")
        print(f"---Total training time: {time.time()-start_time}")




    print("...Done!")