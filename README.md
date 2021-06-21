

# FailBERT

Dyck and Parity were proved theoretically that a transformer architecture could not model them. These limitations can be a drawback when applying transformers-based models to specific natural language tasks involving reasoning capabilities. We designed two different natural datasets to simulate the Dyck-2 and the Parity tasks.

## Installation

Clone this repository

```bash
git clone https://github.com/ChadiHelwe/FailBERT.git
pip install -r requirements.txt
```

## Run Paper Experiments
### Natural Dyck-2 Task (Cake Task)


```bash
python run_natural_dyck_2_experiment.py
```
### Natural Parity Task (Light Switch Task)

```bash
python run_natural_parity_experiment.py
```

## Run Experiments From Scratch

### Natural Dyck-2 Task (Cake Task)

#### Create Dataset
```bash
python run_create_natural_dyck_2_dataset.py  create-natural-dyck-2-dataset  --path_dyck_2_dataset data/dyck_2/dyck_2_test_10.txt --path_natural_dyck_2_dataset data/natural_dyck_2/natural_dyck_2_test_10.csv
```

#### Training a New Model or Download the Pretrained Model

##### Training a New Model

```bash
python run_natural_dyck_2.py train-model
```

#### Download the Pretrained Model

```bash
python run_natural_dyck_2.py download-pretrained-model
```

#### Testing Model

```bash
python run_natural_dyck_2.py test-model
```

### Natural Parity Task (Light Switch Task)

#### Create Dataset

```bash
>python run_create_natural_parity_dataset.py create-natural-parity-dataset --path_natural_parity_dataset data/natural_parity/test1.csv --min_range_length 21 --max_range_length 40 --min_nbr_switch_operation 16 --max_nbr_switch_operation 20
```

#### Balance Dataset

```bash
python run_utils.py create-equally-distributed-natural-dataset --path_dataset data/natural_parity/test1.csv --path_equally_distributed_dataset data/natural_parity/test1.csv --limit True --nbr_instances 2500
```


#### Training a New Model or Download the Pretrained Model

#### Training a New Model

```bash
python run_natural_parity.py train-model
```
#### Download the Pretrained Model

```bash
python run_natural_parity.py download-pretrained-model
```



#### Testing Model

```bash
python run_natural_parity.py test-model
```
