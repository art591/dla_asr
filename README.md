# ASR project barebones

## Installation guide


```shell
pip install -r ./requirements.txt
```

## Best results

| Model  | Librispeech test-clean-100 WER|
| ------------- | ------------- |
| DeepSpeech with Layer Norm | 0.19  |

## Testing best model

Firstly, run the scipt ``` loading_data.sh ``` to load the best model's checkpoint and pretrained language model for beam search.

Then run the following scipt to get CER/WER and model's predictions
```shell
python3 test.py -c config_datasphere.json --resume model_best.pth --output test_reproduced_output.json
```
