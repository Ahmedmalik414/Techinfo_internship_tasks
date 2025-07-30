from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model("my_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    

import numpy as np
import time

text = 'When it grew'
max_len= 17

for i in range(20):

  tokenize_text = tokenizer.texts_to_sequences([text])[0]

  padded_input_text = pad_sequences([tokenize_text],maxlen = max_len-1,padding = 'pre')

  pos = np.argmax(model.predict(padded_input_text))

  for word,index in tokenizer.word_index.items():
    if index == pos:
      text = text + " " + word
      print(text)
      time.sleep(2)    
