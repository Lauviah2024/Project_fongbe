from typing import Union

from fastapi import FastAPI
# from flask import Flask
from deep_translator import GoogleTranslator

from transformers import VitsModel, AutoTokenizer
import torch

import scipy.io.wavfile
import numpy as np
from IPython.display import Audio
from pydantic import BaseModel


class Item(BaseModel):
    text: str

app = FastAPI()
model = VitsModel.from_pretrained("facebook/mms-tts-fon")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fon")

@app.get("/")
def read_root():
    translated = GoogleTranslator(source='fr', target='fon').translate("Bonsoir. Comment vas-tu ?")
    text = translated
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
   
    output = output.cpu()
    data_np = output.numpy()
    data_np_squeezed = np.squeeze(data_np)
    scipy.io.wavfile.write("output.wav", rate=model.config.sampling_rate, data=data_np_squeezed)
    

    return {"data": text, "status": 200}




@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/create_text/{item}")
def createText(item: str):
    print(item)
    untranslated = item  
    translated = GoogleTranslator(source='fr', target='fon').translate(untranslated)
    text = translated

    return {"data": text, "status": 200}