Data and Code for paper *Outlining and Filling: Hierarchical Query Graph Generation for Answering Complex Questions over Knowledge Graph* is available for research purposes.

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

* Download our used [Freebase](https://drive.google.com/file/d/15Ux-zn1xYEh-iVFHudHc6044NYWRfcgN/view?usp=sharing) and [DBpedia](https://drive.google.com/file/d/15Ux-zn1xYEh-iVFHudHc6044NYWRfcgN/view?usp=sharing). Both of them only contain English. Download and install [Virtuoso](https://virtuoso.openlinksw.com) to conduct SPARQL query service.

* Download [Glove Embedding](http://nlp.stanford.edu/data/glove.42B.300d.zip) and put `glove.42B.300d.txt` under `./data/` directory.

### Running Code

#### 1. Training for HGNet
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
The path format of the trained model is `./runs/RUN_ID/checkpoints/best_snapshot_epoch_xx_best_val_acc_xx_model.pt`


#### 2. Testing for HGNet
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
Before testing, need to train the model first and set the following hyperparameters 
```bash
--cpt your_trained_model_path
--kb_endpoint your_sparql_service_ip
```
