# HydraSwin: Multi-Task E-commerce Cataloging via Vision Transformers and Neuro-Symbolic Logic

**Language:** [English](README.md) | [Tiếng Việt](README_VI.md)

[![Python
3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch
2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/VNICT-2026-brightgreen.svg)](https://vnict.org/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official PyTorch implementation** of the paper:\
> **"An Empirical Benchmark of Lightweight CNNs and Vision Transformers
> for Multi-Task E-commerce Cataloging"**\
> *Published in the Proceedings of the 29th National Conference on
> Selected Information and Communications Technology Issues (VNICT
> 2026), Hanoi, Vietnam.*

------------------------------------------------------------------------

## 📌 Overview

Automating product cataloging in e-commerce requires both high
classification accuracy and low inference latency. This repository
introduces **HydraSwin**, an end-to-end multi-task vision framework
built around a shared hierarchical **Swin-Tiny Transformer** backbone
and six parallel classification heads.

To address extreme catalog sparsity and long-tail class imbalance,
HydraSwin combines an **Inverse-Frequency Loss** with a **42-rule
Neuro-Symbolic Filter**. The filter enforces taxonomic and seasonal
constraints to resolve contradictory predictions.

### Model Architecture

``` text
                       ┌───────────────┐
                       │  Input Image  │ (224 x 224 x 3)
                       └───────┬───────┘
                               │
                ┌──────────────▼──────────────┐
                │      Swin-Tiny Backbone     │ (Shared Latent Dim: 768)
                │   (W-MSA + SW-MSA Attention) │
                └──────────────┬──────────────┘
                               │
      ┌──────────┬──────────┬───┴──────┬──────────┬──────────┐
      ▼          ▼          ▼          ▼          ▼          ▼
  [Gender]  [MasterCat]  [SubCat]  [Article]  [Colour]   [Season]
   (5 cls)     (7 cls)    (45 cls)  (142 cls)  (46 cls)   (4 cls)
      │          │          │          │          │          │
      └──────────┴──────────┴───┬──────┴──────────┴──────────┘
                               │
                ┌──────────────▼──────────────┐
                │  Neuro-Symbolic Rule Filter │ (Ontological Constraints)
                └──────────────┬──────────────┘
                               ▼
                    [Verified Product Tags]
```

------------------------------------------------------------------------

## 📊 Benchmark Results

The models were evaluated on the **Kaggle Fashion Product Images**
dataset using a sequestered 10% test split (4,444 images). Measurements
were collected with CPU execution and batch size 1.

  --------------------------------------------------------------------------------------------------------------
  Model               Backbone                       Parameters  FLOPs (G)     Latency   Throughput   Mean Macro
                                                            (M)                   (ms)        (FPS)           F1
  ------------------- ---------------------------- ------------ ---------- ----------- ------------ ------------
  **MobileNetV3**     `mobilenet_v3_large_100`         **5.48**   **0.44**   **24.13**     **41.4**       0.9789

  **ConvNeXt-Tiny**   `convnext_tiny`                     28.59       4.46       86.99         11.5   **0.9803**

  **HydraResNet**     `resnet50`                          25.56       4.12      123.91          8.1       0.8628

  **HydraSwin         `swin_tiny_patch4_window7`          28.29       4.50      126.10          7.9   **0.9600**
  (Ours)**                                                                                          

  **ViT-Tiny**        `vit_tiny_patch16_224`               5.72       1.26       48.50         20.6       0.9411
  --------------------------------------------------------------------------------------------------------------

### Key Findings and Trade-offs

1.  **Edge deployment:** MobileNetV3 delivers the highest efficiency in
    this benchmark, with 24.13 ms latency and 41.4 FPS using 5.48M
    parameters.
2.  **Global context:** HydraSwin uses hierarchical shifted-window
    attention to model non-local relationships across product
    attributes.
3.  **Neuro-symbolic correction:** The rule filter was triggered on 148
    test samples and achieved 91.89% accuracy. It improved Season Macro
    F1 from 0.8910 to 0.9852.

------------------------------------------------------------------------

## 📂 Repository Structure

``` text
HydraSwin-VNICT2026/
├── models/
│   ├── hydraswin.py          # HydraSwin multi-head architecture (Swin-Tiny)
│   └── baselines.py          # MobileNetV3, ConvNeXt-Tiny, ViT baselines
├── loss/
│   └── weighted_loss.py      # Inverse-Frequency class-weighted loss
├── postprocess/
│   └── rule_filter.py        # Neuro-Symbolic 42-rule consistency filter
├── notebooks/
│   └── hydraswin_training_pipeline.ipynb # Original training and evaluation notebook
├── benchmark_results/
│   ├── benchmark_efficiency_results.csv
│   └── rule_ablation_results.csv
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md
```

------------------------------------------------------------------------

## 📖 Citation

If you use this code or research in your work, please cite the VNICT
2026 paper:

``` bibtex
@inproceedings{nguyen2026hydraswin,
  title={An Empirical Benchmark of Lightweight CNNs and Vision Transformers for Multi-Task E-commerce Cataloging},
  author={Nguyen, Bao T. and Nguyen, Dang D. and Duong, Hoc T.},
  booktitle={Proceedings of the 29th National Conference on Selected Information and Communications Technology Issues (VNICT 2026)},
  year={2026},
  address={Hanoi, Vietnam}
}
```

------------------------------------------------------------------------

## 👨‍💻 Authors

-   **Dang D. Nguyen** --- Institute of Intelligent & Interactive
    Technologies, University of Economics Ho Chi Minh City (UEH).\
    **Email:** <nguyendidang1207@gmail.com> \| **ORCID:**
    [0009-0003-1971-8258](https://orcid.org/0009-0003-1971-8258)
-   **Bao T. Nguyen** --- University of Economics Ho Chi Minh City
    (UEH).
-   **Hoc T. Duong** --- University of Economics Ho Chi Minh City (UEH).

------------------------------------------------------------------------

## 📄 License

This project is released under the [MIT License](LICENSE).