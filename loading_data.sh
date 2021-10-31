#!/bin/bash

echo "Loading best model checkpoint"
gdown https://drive.google.com/uc?id=1v1suR5DCNICBf0rzyAwOz4VfDpyi1S6S --output model_best.pth
echo "Loading language model"
gdown https://drive.google.com/uc?id=1pkf-FUKc_1KUN3cfdKgu_csSGqoNPEqr --output lm.binary
