from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pickle
import librosa
import soundfile as sf
import numpy as np
from API.prepoc import extract_features
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load pre-trained model
model_path =  os.path.join("API","model.pkl")
with open(model_path, 'rb') as f:
    app.state.model = pickle.load(f)


@app.post('/upload_wav_file')
def receive_file(file: bytes = File(...)):
    
    with open('API/temp/audio.wav','wb') as audio:  
        audio.write(file)        
        audio.close()  
    
    return 'got it'

@app.get('/predict/')
async def predict():

    # Save audio file to temporary folder
    #with open(f'API/temp/audio.wav', 'wb') as f:
    #   f.write(await file.read())

    #audio = extract_features('API/temp/audio.wav')
    audio_file = 'API/temp/audio.wav'
    
    audio = extract_features(audio_file)


    # predict the class using pre-trained model
    y_pred = app.state.model.predict(audio)

    return {
            "Prediction": int(y_pred[0])
            }


@app.get("/")
def index():
    return {'is it working ?': 'Yes'}