# Analisi del sentiment

Questo esempio addestra un classificatore logistico `ml_core` sulle recensioni
contenute in `recensioni_train.csv` e poi consente di classificare opinioni
inserite da terminale.

Il CSV contiene 1.000 righe bilanciate: 500 positive e 500 negative. Le 52
recensioni già presenti sono conservate; le altre 948 sono esempi sintetici
originali, con la provenienza indicata nella colonna `source` (`existing` o
`synthetic`). Il 99,5% ottenuto dal holdout corrente misura soprattutto la
capacità di riconoscere questi esempi sintetici: non è una stima affidabile
dell'accuratezza su recensioni reali. Per quella servono dati reali e separati
da quelli usati per addestrare il modello.

## Requisiti

La libreria `machineLearning` richiede Python 3.11 (`>=3.11,<3.12`) e NumPy.
Usa quindi un ambiente Python 3.11 e installa o compila prima `machineLearning`.
Se i due progetti sono cartelle sorelle, gli script cercano automaticamente
anche il modulo compilato nelle directory `machineLearning/src` e
`machineLearning/build`.

## Esecuzione

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install numpy

cd ../machineLearning
python -m pip install -e .
cd ../sentiment_analysis

python train_sentiment.py recensioni_train.csv
python predict_sentiment.py
```

L'addestramento salva il modello e il vocabolario in `models/`, ignorata da
Git. Per addestrare con un limite diverso di termini:

```bash
python train_sentiment.py recensioni_train.csv --features 8000
```
