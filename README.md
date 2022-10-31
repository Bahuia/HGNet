Data and Code for paper [*Outlining and Filling: Hierarchical Query Graph Generation for Answering Complex Questions over Knowledge Graphs*](https://arxiv.org/abs/2111.00732) accepted by IEEE Transactions on Knowledge and Data Engineering, is available for research purposes.

### Results
We apply three KGQA benchmarks to evaluate our approach, ComplexWebQuestions ([Talmor and Berant, 2018](https://aclanthology.org/N18-1059.pdf)), LC-QuAD ([Trivedi et al., 2017](https://link.springer.com/chapter/10.1007/978-3-319-68204-4_22)), and WebQSP ([Yih et al., 2016](https://aclanthology.org/P16-2033/)).

| **Dataset**         | Structure Acc. | Query Graph Acc.|  Precision | Recall | F1-score | Hit@1 |
| ------------------- | :------------: | :-------------: | :--------: | :----: | :------: | :---: |
|ComplexWebQuestions  |  66.96         | 51.68           | 65.27      | 68.44  |  64.95   | 65.25 |
|ComplexWebQuestions (Bert-base)  |  72.88        | 57.80           | 68.89      | 73.30  |  68.88   | 68.80 |
|LC-QuAD              |  78.00         | 60.90           | 75.82      | 75.22  |  75.10   | 76.00 |
|LC-QuAD (Bert-base)          |  80.70         | 63.50           | 78.92      | 78.14  |  78.13   | 78.70 |
|WebQSP               |  79.91         | 62.63           | 70.22      | 74.38  |  70.61   | 70.37 |
|WebQSP (Bert-base)              |  85.03         | 70.74           | 76.66      | 79.28  |  76.62   | 76.92 |




### Data
* Download and unzip our preprocessed [data](https://1drv.ms/u/s!AjOOZxoN9FBQgQ5zAKdzO06ghJlP?e=vIsvzz) to `./`, you can also running our scripts under `./preprocess` to obtain them again.

* Download GloVe Embedding [glove.42B.300d.txt](http://nlp.stanford.edu/data/glove.42B.300d.zip) and put it to `your_glove_path`.

* Download our vocabulary from [here](https://1drv.ms/u/s!AjOOZxoN9FBQgQ-iIJMRDYgMeLpg?e=AOxfgs). Unzip and put it under `./`. It contains our used SPARQL cache for Execution-Guided strategy.

### Virtuoso SPARQL Service

Both of the KGs we used only contain English triples by removing other languages. 
Download and install [Virtuoso](https://virtuoso.openlinksw.com) to conduct the SPARQL query service for the downloaded Freebase and DBpedia.
You can also download our conducted Virtuoso SPARQL service [virtuoso-opensource](https://1drv.ms/u/s!AjOOZxoN9FBQgQ1VXpmcQ6XTb5aJ?e=6TLp78) and unzip it in another directory. [Here](https://joernhees.de/blog/2015/11/23/setting-up-a-linked-data-mirror-from-rdf-dumps-dbpedia-2015-04-freebase-wikidata-linkedgeodata-with-virtuoso-7-2-1-and-docker-optional/) is a tutorial on how to install Virtuoso and import the knowledge graph into it. 

1. Get root access
2. Edit `virtuoso-opensource/database/virtuoso.ini` and set the property "DirsAllowed" to your path.
3. Execute the following commands to start the service.
```bash
cd virtuoso-opensource/database/
../bin/virtuoso-t -fd
```

### Running Code

#### 1. Training for HGNet
Before training, first set the following hyperparameter in `main_train_cwq.sh`, `main_train_lcq.sh`, and `main_train_wsp.sh`.
```bash
--glove_path your_glove_path
```

Execute the following command for training model on ComplexWebQuestions.
```bash
sh main_train_cwq.sh
```
Execute the following command for training model on LC-QuAD.
```bash
sh main_train_lcq.sh
```
Execute the following command for training model on WebQSP.
```bash
sh main_train_wsp.sh
```
The trained model file is saved under `./runs` directory.  
The path format of the trained model is `./runs/RUN_ID/checkpoints/best_snapshot_epoch_xx_best_val_acc_xx_model.pt`.

#### 2. Training for HGNet with Bert-base

Execute the following command for training model on ComplexWebQuestions.
```bash
sh main_train_plm_cwq.sh
```
Execute the following command for training model on LC-QuAD.
```bash
sh main_train_plm_lcq.sh
```
Execute the following command for training model on WebQSP.
```bash
sh main_train_plm_wsp.sh
```
The trained model file is saved under `./runs` directory.  
The path format of the trained model is `./runs/RUN_ID/checkpoints/best_snapshot_epoch_xx_best_val_acc_xx_model.pt`.

#### 3. Testing for HGNet
Before testing, need to train a model first and set the following hyperparameters in `main_eval_cwq.sh`, `main_eval_lcq.sh`, and `main_eval_wsp.sh`.
```bash
--cpt your_trained_model_path
--kb_endpoint your_sparql_service_ip
```
You can also directly download our trained models from [here](https://1drv.ms/u/s!AjOOZxoN9FBQgQyGJE12dAGbjuYl?e=RTMUrs). Unzip and put it under `./`.

Execute the following command for testing the model on ComplexWebQuestions.
```bash
sh main_eval_cwq.sh
```
Execute the following command for testing the model on 
LC-QuAD.
```bash
sh main_eval_lcq.sh
```
Execute the following command for testing the model on WebQSP.
```bash
sh main_eval_wsp.sh
```

#### 4. Testing for HGNet with Bert-base
Before testing, need to train a model first and set the following hyperparameters in `main_eval_plm_cwq.sh`, `main_eval_plm_lcq.sh`, and `main_eval_plm_wsp.sh`.
```bash
--cpt your_trained_model_path
--kb_endpoint your_sparql_service_ip
```
You can also directly download our trained models from [here](https://1drv.ms/u/s!AjOOZxoN9FBQgQyGJE12dAGbjuYl?e=RTMUrs). Unzip and put it under `./`.

Execute the following command for testing the model on ComplexWebQuestions.
```bash
sh main_eval_plm_cwq.sh
```
Execute the following command for testing the model on 
LC-QuAD.
```bash
sh main_eval_plm_lcq.sh
```
Execute the following command for testing the model on WebQSP.
```bash
sh main_eval_plm_wsp.sh
```
