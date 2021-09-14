

# FailBERT

Transformers have achieved impressive performance in a wide range of NLP tasks. Yet, it has been [proven](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00306/43545/Theoretical-Limitations-of-Self-Attention-in) that transformers cannot model two types of languages: the Dyck language (which consists of bracketed expressions) and the Parity language (which consists of sequences of 0's and 1's with an even number of 1's). In our work, we show that this limitation can impact the performance of transformers on reasoning in natural language. We designed two different natural datasets to simulate the Dyck and the Parity tasks.

> Chadi Helwe, Chloé Clavel, Fabian M. Suchanek:  
> [	“Reasoning with Transformer-based Models: Deep Learning, but Shallow Reasoning”](https://suchanek.name/work/publications/akbc-2021-reasoning.pdf)  
> Automated Knowledge Base Construction (AKBC), 2021

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
python run_create_natural_dyck_2_dataset.py  create-natural-dyck-2-dataset  --path_dyck_2_dataset data/dyck_2/dyck_2_test_1.txt --path_natural_dyck_2_dataset data/natural_dyck_2/natural_dyck_2_test_1.csv
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
python run_create_natural_parity_dataset.py create-natural-parity-dataset --path_natural_parity_dataset data/natural_parity/test1.csv --min_range_length 21 --max_range_length 40 --min_nbr_switch_operation 16 --max_nbr_switch_operation 20
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
