#!/usr/bin/env sh

if [ ! -f data/handwriting-recognition.zip ]; then
  curl -L -o data/handwriting-recognition.zip\
    https://www.kaggle.com/api/v1/datasets/download/landlord/handwriting-recognition
fi

unzip data/handwriting-recognition.zip -d data/handwriting-dataset
rm data/handwriting-recognition.zip