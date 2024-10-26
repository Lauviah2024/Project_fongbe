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






@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/create_text")
def createText(item: Item):
    print(item)
    untranslated = item.text

    
    translated = GoogleTranslator(source='fr', target='fon').translate(untranslated)
    text = translated
    print(text)
    inputs = tokenizer(text, return_tensors="pt")

    print(inputs)
    with torch.no_grad():
        output = model(**inputs).waveform
    output = output.cpu()
    data_np = output.numpy()
    data_np_squeezed = np.squeeze(data_np)
    # scipy.io.wavfile.write("output.wav", rate=model.config.sampling_rate, data=data_np_squeezed)
    print (output)

    return {"data": data_np_squeezed, "status": 200}