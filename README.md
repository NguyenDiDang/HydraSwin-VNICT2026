# HydraSwin: Multi-Task E-commerce Cataloging via Vision Transformers and Neuro-Symbolic Logic

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/VNICT-2026-brightgreen.svg)](https://vnict.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official PyTorch Implementation** for the paper:  
> **"An Empirical Benchmark of Lightweight CNNs and Vision Transformers for Multi-Task E-commerce Cataloging"**  
> *Published in The 29th National Conference on Selected Information and Communications Technology Issues (VNICT 2026), Hanoi, Vietnam.*

---

## 📌 Overview

Automating product categorization in e-commerce requires both high accuracy and rapid inference latency. This repository presents **HydraSwin**, an end-to-end multi-task vision framework leveraging a shared hierarchical **Swin-Tiny Transformer** backbone with six concurrent classification heads.

To tackle extreme catalog sparsity and long-tail class imbalance, **HydraSwin** integrates an **Inverse-Frequency Loss** alongside a **42-rule Neuro-Symbolic Filter** that systematically resolves taxonomic and seasonal prediction contradictions.

```
                      ┌───────────────┐
                      │  Input Image  │ (224 x 224 x 3)
                      └───────┬───────┘
                              │
               ┌──────────────▼──────────────┐
               │    Swin-Tiny Backbone       │ (Shared Latent Dim: 768)
               │ (W-MSA + SW-MSA Attention)  │
               └──────────────┬──────────────┘
                              │
     ┌──────────┬──────────┬──┴───────┬──────────┬──────────┐
     ▼          ▼          ▼          ▼          ▼          ▼
[Gender]  [MasterCat]  [SubCat]  [Article]  [Colour]   [Season]
 (5 cls)    (7 cls)    (45 cls)  (142 cls)  (46 cls)   (4 cls)
     │          │          │          │          │          │
     └──────────┴──────────┴──┬───────┴──────────┴──────────┘
                              │
               ┌──────────────▼──────────────┐
               │ Neuro-Symbolic Rule Filter  │ (Ontological Constraints)
               └──────────────┬──────────────┘
                              ▼
                  [Verified Product Tags]
```

---

## 📊 Benchmark Results

Evaluated on the Kaggle Fashion Product Images dataset (sequestered 10% test split, 4,444 images) with CPU execution at batch size 1:

| Model | Backbone | Parameters (M) | FLOPs (G) | Latency (ms) | Throughput (FPS) | Mean Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3** | `mobilenet_v3_large_100` | **5.48** | **0.44** | **24.13** | **41.4** | 0.9789 |
| **ConvNeXt-Tiny** | `convnext_tiny` | 28.59 | 4.46 | 86.99 | 11.5 | **0.9803** |
| **HydraResNet** | `resnet50` | 25.56 | 4.12 | 123.91 | 8.1 | 0.8628 |
| **HydraSwin (Ours)** | `swin_tiny_patch4_window7` | 28.29 | 4.50 | 126.10 | 7.9 | **0.9600** |
| **ViT-Tiny** | `vit_tiny_patch16_224` | 5.72 | 1.26 | 48.50 | 20.6 | 0.9411 |

### Key Findings & Trade-offs:
1. **Edge Deployment:** **MobileNetV3** achieves the highest efficiency (24.1 ms, 41.4 FPS) with minimal 5.48M parameters.
2. **Global Context:** **HydraSwin** eliminates inductive sliding-window bias, modeling non-local cross-attribute correlations through hierarchical shifted-window attention.
3. **Neuro-Symbolic Correction:** The rule filter triggered across 148 test samples with an **accuracy of 91.89%**, lifting the Season Macro F1 from **0.8910 to 0.9852**.

---

## 📂 Repository Structure

```
HydraSwin-VNICT2026/
├── models/
│   ├── hydraswin.py          # HydraSwin multi-head architecture (Swin-Tiny)
│   └── baselines.py          # MobileNetV3, ConvNeXt-Tiny, ViT baselines
├── loss/
│   └── weighted_loss.py      # Inverse-Frequency class-weighted loss
├── postprocess/
│   └── rule_filter.py        # Neuro-Symbolic 42-rule consistency filter
├── notebooks/
│   └── hydraswin_training_pipeline.ipynb # Original training & evaluation notebook
├── benchmark_results/
│   ├── benchmark_efficiency_results.csv
│   └── rule_ablation_results.csv
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md
```

---

## 📖 Citation

If you find this code or research useful, please cite our VNICT 2026 paper:

```bibtex
@inproceedings{nguyen2026hydraswin,
  title={An Empirical Benchmark of Lightweight CNNs and Vision Transformers for Multi-Task E-commerce Cataloging},
  author={Nguyen, Bao T. and Nguyen, Dang D. and Duong, Hoc T.},
  booktitle={Proceedings of the 29th National Conference on Selected Information and Communications Technology Issues (VNICT 2026)},
  year={2026},
  address={Hanoi, Vietnam}
}
```

---

## 👨‍💻 Authors

* **Dang D. Nguyen** — Institute of Intelligent & Interactive Technologies, University of Economics Ho Chi Minh City (UEH).  
  *Email:* [nguyendidang1207@gmail.com](mailto:nguyendidang1207@gmail.com) | *ORCID:* [0009-0003-1971-8258](https://orcid.org/0009-0003-1971-8258)
* **Bao T. Nguyen** — University of Economics Ho Chi Minh City (UEH).
* **Hoc T. Duong** — University of Economics Ho Chi Minh City (UEH).
