# bioModeling - Supervised Random Walk (SRW)

Follow the steps below to successfully run the code.

## 1. Install Environment
```
conda create -n srw python=3.10 -y
conda activate srw
pip install -r requirements.txt
```

## 2. Prepare Config File
Each dataset needs a YAML config in the configs/ folder. <br>
Example is in configs/BRCA_small.yaml


## 3. Run Training
```
python train.py --config configs/BRCA_small.yaml
python train.py --config configs/GBM.yaml
```