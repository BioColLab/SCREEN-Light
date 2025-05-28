# Streamlined SCREEN for catalytic residue prediction
## Background

This repository offers a streamlined, faster version of SCREEN with fewer software dependencies. 

Generating evolutionary features using PSI-BLAST and HMMER is time-consuming, but as shown in our ablation study, removing PSSM and HMM features causes only a minor performance drop. Thus, for practical large-scale use, this simplified SCREEN balances speed and accuracy effectively.

## Installation
We recommend setting up a dedicated virtual environment with the specified versions to ensure reproducibility.
You can create and activate the environment using Anaconda:
```
conda create --name SCREEN_env python=3.6
conda activate SCREEN_env
conda install pytorch==1.7.0 torchvision==0.4.0 torchaudio==0.7.0 -c pytorch 
pip install -r requirements.txt
```
Additionally, SCREEN utilizes the protT5 pre-trained language model for enzyme feature extraction (referring to https://github.com/agemagician/ProtTrans)

The language model protT5  can be installed via

```
pip install transformers
pip install sentencepiece
```

## Predicting catalytic residues 

To predict catalytic residues, only the target enzyme sequence in FASTA format is required. Example usage:
```
python predict_SCREEN.py 
```

## Training from scratch

```
python train_SCREEN.py
```
Note: 
Training datasets are located in the Dataset/training_data directory and include:

training_id_withEC.txt: A list of enzyme IDs with EC number annotations.

training_label.txt: Contains enzyme sequences and their catalytic residue annotations.

## Contact
For any question and comment regarding the code, please reach out to
tong.pan@monash.edu



