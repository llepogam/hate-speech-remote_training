import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time

import mlflow
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, GRU, LSTM, GlobalMaxPooling1D, Dropout, Input
from tensorflow.keras.layers import TextVectorization

import spacy
from spacy.lang.en.stop_words import STOP_WORDS


#------------------ Let's build the function that will be used in the training------------------

def read_dataframe():
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    df = pd.read_csv("hf://datasets/christophsonntag/OLID/" + splits["train"])   
    return df

def clean_dataframe(df):
    selected_columns = ['tweet', 'subtask_a']
    df_simplified = df[selected_columns]
    df_simplified.loc[:, 'target'] = 0
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

def save_preprocessed_data(df):
    df.to_csv("preprocessed.csv", index=True)


def get_preprocessed_data():

    df = pd.read_csv("preprocessed.csv")    
    df["text_clean"] = df["text_clean"].astype(str)   
    df["target"] = pd.to_numeric(df["target"], errors='coerce')
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
                    GRU(units=128, return_sequences=True),
                    GRU(units=128, return_sequences=True),
                    GlobalMaxPooling1D(),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1, activation="sigmoid")
    ]) 


    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  
        metrics=['accuracy']
    )
    return model, vectorizer


#------------------ Let's here write the flow of the MLFlow run------------------

if __name__ == "__main__":

    experiment_name = "hate_speech_detection"
    mlflow.set_experiment(experiment_name)

    mlflow.tensorflow.autolog()  


    with mlflow.start_run():

        start_time = time.time()

        print("reading dataframe...")
        df = read_dataframe()
        print("dataframe read...")

        print("cleaning dataframe...")
        df_clean = clean_dataframe(df)
        print("dataframe cleaned...")        

        print("preprocessing dataframe...")
        #df_preprocessed = preprocess_dataframe(df_clean)
        #save_preprocessed_data(df_preprocessed)
        df_preprocessed = get_preprocessed_data()
        print("dataframe preprocessed...")  

        #Parameters
        input_shape_length = 100
        embedding_dim = 32

        print("splitting the data...")
        xtrain, xval, ytrain, yval = train_test_split(df_preprocessed["text_clean"], df_preprocessed["target"], test_size=0.2)

        train = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        val = tf.data.Dataset.from_tensor_slices((xval, yval))
        train_batch = train.shuffle(len(train)).batch(64)
        val_batch = val.shuffle(len(val)).batch(64)
        print("data split..")        


        print("building model...")
        model,vectorizer = build_model(input_shape_length,embedding_dim)

        # Adapt the vectorizer to the training data
        print("adapting vectorizer...")
        vectorizer.adapt(train.map(lambda x, y: x))

        print("training model...")
        history = model.fit(train_batch, epochs = 3, validation_data=val_batch)

        # Log metrics for each epoch
        for epoch in range(len(history.history['loss'])):
            for metric_name in history.history.keys():
                mlflow.log_metric(metric_name, history.history[metric_name][epoch], step=epoch)

        # Infer and log the model signature
        input_sample = next(iter(train_batch.take(1)))[0]
        predicted_sample = model.predict(input_sample)
        signature = infer_signature(input_sample.numpy(), predicted_sample)
        mlflow.tensorflow.log_model(model, artifact_path="hate_speech_detection", signature=signature)
 
        #Make predictions
        y_pred_probs = model.predict(xval.to_numpy())
        y_pred = (y_pred_probs > 0.5).astype(int) 

        # Compute the confusion matrix
        cm = confusion_matrix(yval, y_pred)  # y_test is the true labels for X_test

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Hate", "Hate"])
        disp.plot()
        plt.savefig('confusion_matrix.png') 

        mlflow.log_artifact('confusion_matrix.png')

        precision = precision_score(yval, y_pred)
        recall = recall_score(yval, y_pred)
        f1 = f1_score(yval, y_pred)

        mlflow.log_param("precision", precision)
        mlflow.log_param("recall", recall)
        mlflow.log_param("f1", f1)


        print("...Done!")
        print(f"---Total training time: {time.time()-start_time}")




    print("...Done!")