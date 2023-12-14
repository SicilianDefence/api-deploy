from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import keras.models
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle

nltk.download('punkt')
nltk.download('stopwords')

with open('token_fix.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('token_hate.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)

loaded_model = keras.models.load_model('ai_fix.h5')
loaded_model2 = keras.models.load_model('ai_hate.h5')

def preprocessing_teks(dataset):
  dataset = dataset.lower() #CaseFolding
  dataset = dataset.replace(r'[^\w\s]+', '') # Penghapusan Tanda Baca
  dataset = nltk.word_tokenize(dataset)
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  hasil = []
  for j in dataset:
    stemmed=stemmer.stem(j)
    hasil.append(stemmed)
  return hasil

class Item(BaseModel):
    query: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_hoax(item: Item):
    teks = item.query

    # vectorize the user's query and make a prediction
    teks_preprocess = preprocessing_teks(teks)
    prepad = tokenizer.texts_to_sequences([teks_preprocess])
    prepad2 = pad_sequences(prepad, maxlen=100, padding='pre')

    pred_text = "Hoax" if(np.round(loaded_model.predict(prepad2)) == 0) else "Fakta"

    # create JSON object
    output = {'prediction': pred_text}

    return output

@app.post("/hate")
async def predict_hate(item: Item):
    teks = item.query

    # vectorize the user's query and make a prediction
    teks_preprocess = preprocessing_teks(teks)
    prepad = tokenizer2.texts_to_sequences([teks_preprocess])
    prepad2 = pad_sequences(prepad, maxlen=100, padding='pre')

    pred_text = "Hate Speech" if(np.round(loaded_model.predict(prepad2)) == 0) else "Netral"

    # create JSON object
    output = {'prediction': pred_text}

    return output