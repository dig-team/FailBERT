# FailBERT

Dyck and Parity were proved theoretically that a transformer architecture could not model them. These limitations can be a drawback when applying transformers-based models to specific natural language tasks involving reasoning capabilities. We designed two different natural datasets to simulate the Dyck-2 and the Parity tasks.

## Installation

Clone this repository

```bash
git clone https://github.com/ChadiHelwe/FailBERT.git
pip install -r requirements.txt
```

## Usage

### Natural Dyck-2 Task

#### Create Dataset

```bash
python run_natural_dyck_2_dataset.py create-natural-dyck-2-dataset
```

#### Balance Dataset

```bash
python run_utils.py create-equally-distributed-natural-dataset
```

#### Split Dataset

```bash
python run_utils.py split-natural-dataset
```

#### Download Pretrained Model

```bash
python run_natural_dyck_2.py download-pretrained-model
```

#### Training a New Model

```bash
python run_natural_dyck_2.py train-model
```

#### Testing Model

```bash
python run_natural_dyck_2.py test-model
```

### Natural Parity Task

#### Create Dataset

```bash
python run_natural_parity_dataset.py create-natural-parity-dataset
```

#### Balance Dataset

```bash
python run_utils.py create-equally-distributed-natural-dataset
```

#### Split Dataset

```bash
python run_utils.py split-natural-dataset
```

#### Download Pretrained Model

```bash
python run_natural_parity.py download-pretrained-model
```


#### Training a New Model

```bash
python run_natural_parity.py train-model
```

#### Testing Model

```bash
python run_natural_parity.py test-model
```
