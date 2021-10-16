Data and Code for paper *Outlining and Filling: Hierarchical Query Graph Generation for Answering Complex Questions over Knowledge Graph* is available for research purposes.

### Results
We apply three KGQA benchmarks to evaluate our approach, ComplexWebQuestions ([Talmor and Berant, 2018](https://aclanthology.org/N18-1059.pdf)), LC-QuAD ([Trivedi et al., 2017](https://link.springer.com/chapter/10.1007/978-3-319-68204-4_22)), and WebQSP ([Yih et al., 2016](https://aclanthology.org/P16-2033/)).

| **Dataset**         | Structure Acc. | Query Graph Acc.|  Precision | Recall | F1-score | Hit@1 |
| ------------------- | :------------: | :-------------: | :--------: | :----: | :------: | :---: |
|ComplexWebQuestions  |  66.96         | 51.68           | 65.27      | 68.44  |  64.95   | 65.25 |
|LC-QuAD              |  78.00         | 60.90           | 75.82      | 75.22  |  75.10   | 76.00 |
|WebQSP               |  79.91         | 62.63           | 70.22      | 74.38  |  70.61   | 70.37 |


### Requirements
* Python == 3.7.0
* cudatoolkit == 10.1.243
* cudnn == 7.6.5
* six == 1.15.0
* torch == 1.4.0
* transformers == 4.9.2
* numpy == 1.19.2
* SPARQLWrapper == 1.8.5
* rouge_score == 0.0.4
* filelock == 3.0.12
* nltk == 3.6.2
* absl == 0.0
* dataclasses == 0.6
* datasets == 1.9.0
* jsonlines == 2.0.0
* python_Levenshtein == 0.12.2
* [Virtuoso](https://virtuoso.openlinksw.com) SPARQL query service

### Data
* Download and unzip our preprocessed [data](https://drive.google.com/file/d/15Ux-zn1xYEh-iVFHudHc6044NYWRfcgN/view?usp=sharing) to `./`, you can also running our scripts under `./preprocess` to obtain them again.

* Download our used [Freebase](https://drive.google.com/file/d/1Yh5eXX13mTetFec_49CaLkjDw62DF8f9/view?usp=sharing) and [DBpedia](https://drive.google.com/file/d/17TRlj8a34IEo686nnKHTewZEg4aMrYUe/view?usp=sharing). Both of them only contain English triples by removing other languages. Download and install [Virtuoso](https://virtuoso.openlinksw.com) to conduct the SPARQL query service for the downloaded Freebase and DBpedia. [Here](https://joernhees.de/blog/2015/11/23/setting-up-a-linked-data-mirror-from-rdf-dumps-dbpedia-2015-04-freebase-wikidata-linkedgeodata-with-virtuoso-7-2-1-and-docker-optional/) is a tutorial on how to install Virtuoso and import the knowledge graph into it.

* Download GloVe Embedding [glove.42B.300d.txt](http://nlp.stanford.edu/data/glove.42B.300d.zip) and put it to `your_glove_path`.

* Download our vocabulary from [here](https://drive.google.com/file/d/1vKqs6r96KTk34-9xz8QS_Dz56OCRNV-U/view?usp=sharing). Unzip and put it under `./`. It contains our used SPARQL cache for Execution-Guided strategy.

### Running Code

#### 1. Training for HGNet
Before training, first set the following hyperparameter in `train_cwq.sh`, `train_lcq.sh`, and `train_wsp.sh`.
```bash
--glove_path your_glove_path
```

Execute the following command for training model on ComplexWebQuestions.
```bash
sh train_cwq.sh
```
Execute the following command for training model on LC-QuAD.
```bash
sh train_lcq.sh
```
Execute the following command for training model on WebQSP.
```bash
sh train_wsp.sh
```
The trained model file is saved under `./runs` directory.  
The path format of the trained model is `./runs/RUN_ID/checkpoints/best_snapshot_epoch_xx_best_val_acc_xx_model.pt`.


#### 2. Testing for HGNet
Before testing, need to train a model first and set the following hyperparameters in `eval_cwq.sh`, `eval_lcq.sh`, and `eval_wsp.sh`.
```bash
--cpt your_trained_model_path
--kb_endpoint your_sparql_service_ip
```
You can also directly download our trained models from [here](https://drive.google.com/file/d/11IVPgKtVyRcA9Xprb6M_jkqUDDTALwVj/view?usp=sharing). Unzip and put it under `./`.

Execute the following command for testing the model on ComplexWebQuestions.
```bash
sh eval_cwq.sh
```
Execute the following command for testing the model on 
LC-QuAD.
```bash
sh eval_lcq.sh
```
Execute the following command for testing the model on WebQSP.
```bash
sh eval_wsp.sh
```
