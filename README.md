# DaST: Data-Free Substitute Training for Adversarial Attacks

## Paper Reference
This project reproduces and analyzes:

**DaST: Data-free Substitute Training for Adversarial Attacks (CVPR 2020)**  
https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_DaST_Data-Free_Substitute_Training_for_Adversarial_Attacks_CVPR_2020_paper.html  

---

## Overview

This project implements **DaST**, a method for performing **black-box adversarial attacks without any real training data**.

Traditional substitute attacks require:
- real data OR
- a pre-trained model

DaST removes both requirements by generating synthetic data using a GAN and training a substitute model from scratch.

---

## Key Idea

DaST trains a substitute model without real data using three components:

- **Generator (G)** → creates synthetic inputs  
- **Substitute model (D)** → learns to imitate the target model  
- **Target model (T)** → provides labels or probabilities  

### Workflow

1. Generate synthetic data:  
   `X_b = G(z)`
2. Query the target model:  
   `T(X_b)`
3. Train substitute model using:  
   `(X_b, T(X_b))`
4. Use substitute model to generate adversarial examples

---

## Threat Model

DaST operates in a **black-box setting**:

### 1. Label-only (DaST-L)
- Only predicted labels are available

### 2. Probability-only (DaST-P)
- Full output probabilities are available

---

## Method

### Objective (Substitute Model)

Minimize the difference between substitute model and target model:
min D d(T(X_b), D(X_b))


---

### Generator Objective

Maximize disagreement:
max G d(T(X_b), D(X_b))

---

### Key Innovation

#### 1. Multi-branch Generator
- One branch per class
- Ensures coverage across all categories

#### 2. Label-Control Loss
L_C = CE(D(G(z, n)), n)

- Forces generated samples to span all classes
- Prevents mode collapse
- Improves data diversity

---

## Why DaST Works

- Substitute model approximates the target model
- Adversarial examples transfer between models
- No real data is required

---

## Experiments (From Paper)

### MNIST
- DaST-P accuracy: ~97.8%
- DaST-L accuracy: ~83.9%

### Azure (Real-world model)
- Attack success rate up to ~98%

---

## Environment

- Python 3.9  
- PyTorch 1.12.1 + CUDA 11.3  
- torchvision 0.13.1  
- advertorch  

---

## Installation

### 1. Clone repository

```bash
git clone <your-repo-link>
cd DaST
```

### 2. Create virtual environment
```bash
python3 -m venv dast-env
source dast-env/bin/activate
```

### 3. Install dependencies
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install advertorch joblib
```

### Running the Code (HPC - Slurm)
Submit job:
```bash
sbatch run_dast.sh
```

Monitor the job:
```bash
squeue -u $USER
tail -f dast_<JOBID>.log
```

## Conclusion

### Citation
```latex
@inproceedings{zhou2020dast,
  title={DaST: Data-free Substitute Training for Adversarial Attacks},
  author={Zhou, Mingyi et al.},
  booktitle={CVPR},
  year={2020}
}
```

### Author
Karina Larochelle
CSC 592 Final Project
