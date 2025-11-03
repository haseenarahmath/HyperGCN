# HyperGCN — A Multi-Layer Multi-Exit Graph Neural Network to Enhance Hyperspectral Image Classification

**Haseena Rahmath P, Kuldeep Chaurasia, Anika Gupta, and Vishal Srivastava**  
*International Journal of Remote Sensing (2024), Vol. 45, No. 14, pp. 4848–4882*  
DOI: [10.1080/01431161.2024.2370501](https://doi.org/10.1080/01431161.2024.2370501)

---

### Overview
This repository accompanies the *International Journal of Remote Sensing (2024)* paper  
**“HyperGCN – A Multi-Layer Multi-Exit Graph Neural Network to Enhance Hyperspectral Image Classification.”**

HyperGCN introduces a deep, oversmoothing-resistant graph neural network integrated with multiple early-exit branches for adaptive inference in hyperspectral image classification.  
The framework balances spectral–spatial feature learning with computational efficiency, achieving superior accuracy on four benchmark datasets (*Indian Pines*, *Pavia University*, *Salinas*, and *Botswana*).  
All resources and experiments were implemented in **Python / PyTorch**, following the configurations described in the publication.

---

### Main Contributions
1. **Deep GCN Backbone** — Five graph-convolution blocks with residual mapping for spectral–spatial learning.  
2. **Oversmoothing Resistance** — Incorporates PairNorm normalization and skip connections to preserve feature diversity and stabilize deep-layer training.  
3. **Early-Exit Mechanism** — Side branches enable adaptive inference, reducing latency without compromising accuracy.  
4. **Comprehensive Evaluation** — Benchmarked on four hyperspectral datasets using OA, AA, and Cohen’s κ.  
5. **Efficiency–Accuracy Trade-off** — Achieves competitive accuracy while lowering computational cost.  
6. **Generalizable Framework** — Extensible to other spectral–spatial or graph-structured learning problems.

---

### Implementation Details
- **Framework:** PyTorch (Python 3.x)  
- **Optimizer:** Adam  
- **Learning Rate:** 0.005  
- **Epochs:** 1000  
- **Hidden Units:** 64  
- **Dropout:** 0.06 (within convolution blocks), 0.4 (before the final fully connected layer)  
- **Normalization:** PairNorm  
- **Activation:** ReLU  
- **Loss Function:** Weighted cross-entropy combining all exit branches  
- **Inference Threshold:** Confidence range 0.6–0.9  
- **Metrics:** Overall Accuracy (OA), Average Accuracy (AA), and Cohen’s κ  
- **Hardware:** NVIDIA GTX GPU and Intel i5 CPU (2.8 GHz)

---

### Training Strategy
HyperGCN and its multi-exit variant (**M-HyperGCN**) are trained jointly using a weighted cross-entropy objective that aggregates losses from all exit branches.  
Each exit branch is optimized to produce consistent and confident predictions, while gradients from early exits stabilize intermediate layers.  
During inference, the model selects an exit based on a softmax confidence threshold (0.6–0.9), enabling adaptive computation while preserving accuracy.

---

### Datasets
| Dataset | Size | Bands | Classes |
|:--|:--:|:--:|:--:|
| *Indian Pines* | 145 × 145 | 220 | 16 |
| *Pavia University* | 610 × 340 | 103 | 9 |
| *Salinas* | 512 × 217 | 224 | 16 |
| *Botswana* | 1476 × 256 | 242 | 14 |

---

### Citation
```bibtex
@article{RahmathP2024HyperGCN,
  author    = {Haseena Rahmath P and Kuldeep Chaurasia and Anika Gupta and Vishal Srivastava},
  title     = {HyperGCN – a multi-layer multi-exit graph neural network to enhance hyperspectral image classification},
  journal   = {International Journal of Remote Sensing},
  volume    = {45},
  number    = {14},
  pages     = {4848--4882},
  year      = {2024},
  publisher = {Taylor & Francis},
  doi       = {10.1080/01431161.2024.2370501},
  url       = {https://doi.org/10.1080/01431161.2024.2370501},
  eprint    = {https://doi.org/10.1080/01431161.2024.2370501}
}
