# AntiSpoofing Project

## Installation guide 

As usual minimal requirements are provided by standart Kaggle environment, but you also need the following lib

```shell
pip install hydra-core
```

To run the download weight of the model please run the following code

```shell
python download.py
```

Inference:

```shell
python test.py resume="WEIGHTS" test_data_path="DATA"
```

## Details

There was a whole lot of research done on this model during which I managed to significantly improve the score, please refer to the report for more details.

https://wandb.ai/bayesian_god/AS_project/reports/SOTA-Voice-AntiSpoofing-with-RawNet2--Vmlldzo2MjQzNDgw

