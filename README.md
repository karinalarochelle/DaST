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
- advertorch 0.2.4
- joblib 1.1.0

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
pip install advertorch==0.2.4 joblib==1.1.0
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

The run_dast.sh script executes:
```bash
python -u run_dast.py --dataset=mnist --cuda --workers=1 --niter=50
```
With this command you can modify:
- --dataset -> mnist or fashionmnist
- --niter -> number of epochs
- --workers -> recommended: 1 for HPC

#### Running Evaluation
```bash
python evaluation.py --dataset=mnist --cuda
```
- This generates adversarial examples using FGSM/BIM/PGD
- Uses epsilon values up to 0.26 (max pertubation strength)

---

## Novel Idea
### Motivation
The original DaST paper primarily evaluates the method on datasets such as MNIST and real-world models (e.g., Azure). To explore the generalizability of the approach, I extended the implementation to support the FashionMNIST dataset, which contains more complex and visually diverse images than MNIST.

This experiment aims to evaluate whether DaST can effectively generate adversarial examples in a setting with higher intra-class variability.

### Implementation
To support FashionMNIST, I modified the dataset loading section in dast.py by adding a new condition:
```python
elif opt.dataset == 'fashionmnist':
    testset = torchvision.datasets.FashionMNIST(
        root='dataset/', train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
```
The same model architectures used for MNIST were reused:
- Net_l as the substitute model (netD)
- Net_m as the target model

Additionally, I configured the adversarial attack using: 
- Linf Basic Iterative Attack (PGD-style)
- ε = 0.25
- 200 iterations
This ensures consistency with the MNIST setup while allowing direct comparison across datasets.

### Results
When applying DaST to FashionMNIST, the model achieved:
- Attack success rate: ~93% – 97%
- Substitute model accuracy: ~7% – 11%

Compared to MNIST (which showed lower attack success rates under reduced training), FashionMNIST produced significantly stronger attack performance.

### Observations
- The generator was able to produce highly effective adversarial examples on FashionMNIST
- The substitute model successfully approximated the target model despite no access to real training data
- The high attack success rate suggests strong transferability of adversarial examples

Interestingly, FashionMNIST appeared more vulnerable to this attack than MNIST under the same training constraints.

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
If you have any questions, you can contact me at [klarochelle@uri.edu](url).
