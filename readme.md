# GraphADT: Empowering Interpretable Predictions of Acute Dermal Toxicity
GraphADT is an advanced model designed to accurately predict acute dermal toxicity (ADT) of compounds using innovative techniques such as structure remapping and multi-view graph pooling. By transforming chemical bonds into new nodes and employing multiple views to rank node importance, GraphADT enhances the interpretability of ADT predictions. This approach outperforms existing models by emphasizing critical substructures like functional groups and toxicophores. The model's effectiveness has been validated on public datasets, demonstrating its potential in guiding the development of safer contact drugs.

![GraphADT](models/GraphADT.png)

## 1 File Structure
```shell
├── ana ## Experiments for Analysis
├── create_data.py  ## Create graph Data
├── dataset the overall Dataset
│   ├── Rabbit.csv
│   ├── Rabbit_external.csv
│   ├── Rat.csv
│   └── Rat_external.csv
├── evalution.py
├── Graph_based_interpretability # Shapley Analysis
│   ├── bond_data_0.pkl
│   ├── bond_data_1.pkl
│   ├── data
│   ├── edgeshape0.py
│   ├── edgeshape1.py
│   ├── __pycache__
│   ├── rdkit_heatmaps
│   ├── statistic0.py
│   └── statistic1.py
├── main.py
├── models  
│   ├── model.py ## GraphADT model
│   ├── MVPool
│   └── __pycache__
├── nt_xent.py
├── Readme.md
└── structuralremap_construction ##structuralremap_construction
    ├── data
    ├── __init__.py
    └── __pycache__

```

## 2  Environment Setup
！[Download Conda Environment](/home/dell/mxq/toxic_mol/model/GraphADT/GRAPHADT/environment.yml)
## 2.1 Full Environment Setup
```bash
conda create -n GraphADT python=3.8
conda activate GraphADT
# It is recommended to use the specified versions of PyTorch and CUDA here to meet the dependencies of the uni-mol model later on.
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia 


