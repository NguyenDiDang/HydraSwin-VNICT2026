# HydraSwin: Phân Loại Đa Nhiệm Sản Phẩm Thương Mại Điện Tử Bằng Vision Transformer Và Logic Neuro-Symbolic

**Ngôn ngữ:** [English](README.md) | [Tiếng Việt](README_VI.md)

[![Python
3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch
2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/VNICT-2026-brightgreen.svg)](https://vnict.org/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Mã nguồn triển khai chính thức bằng PyTorch** cho bài báo:\
> **"An Empirical Benchmark of Lightweight CNNs and Vision Transformers
> for Multi-Task E-commerce Cataloging"**\
> *Được xuất bản trong Kỷ yếu Hội nghị Quốc gia lần thứ 29 về một số vấn
> đề chọn lọc của Công nghệ thông tin và Truyền thông (VNICT 2026), Hà
> Nội, Việt Nam.*


------------------------------------------------------------------------

## 📌 Tổng quan

Tự động hóa việc phân loại và gán nhãn sản phẩm trong thương mại điện tử
đòi hỏi mô hình vừa có độ chính xác cao, vừa có độ trễ suy luận thấp.
Repository này giới thiệu **HydraSwin**, một framework thị giác máy tính
đa nhiệm từ đầu đến cuối, sử dụng backbone **Swin-Tiny Transformer**
phân cấp dùng chung và sáu nhánh phân loại song song.

Để xử lý tình trạng dữ liệu danh mục thưa thớt nghiêm trọng và mất cân
bằng lớp theo phân phối long-tail, HydraSwin kết hợp **Inverse-Frequency
Loss** với **bộ lọc Neuro-Symbolic gồm 42 luật**. Bộ lọc áp dụng các
ràng buộc về phân cấp danh mục và mùa vụ nhằm xử lý những dự đoán mâu
thuẫn.

### Kiến trúc mô hình

``` text
                       ┌───────────────┐
                       │  Ảnh đầu vào  │ (224 x 224 x 3)
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
   (5 lớp)     (7 lớp)    (45 lớp)  (142 lớp)  (46 lớp)   (4 lớp)
      │          │          │          │          │          │
      └──────────┴──────────┴───┬──────┴──────────┴──────────┘
                               │
                ┌──────────────▼──────────────┐
                │ Bộ lọc Neuro-Symbolic       │ (Ràng buộc ontology)
                └──────────────┬──────────────┘
                               ▼
                    [Các nhãn sản phẩm đã xác thực]
```

------------------------------------------------------------------------

## 📊 Kết quả Benchmark

Các mô hình được đánh giá trên bộ dữ liệu **Kaggle Fashion Product
Images**, sử dụng tập kiểm thử 10% được giữ riêng (4.444 ảnh). Các phép
đo được thực hiện trên CPU với batch size bằng 1.

  --------------------------------------------------------------------------------------------------------------
  Mô hình             Backbone                       Parameters  FLOPs (G)     Latency   Throughput   Mean Macro
                                                            (M)                   (ms)        (FPS)           F1
  ------------------- ---------------------------- ------------ ---------- ----------- ------------ ------------
  **MobileNetV3**     `mobilenet_v3_large_100`         **5.48**   **0.44**   **24.13**     **41.4**       0.9789

  **ConvNeXt-Tiny**   `convnext_tiny`                     28.59       4.46       86.99         11.5   **0.9803**

  **HydraResNet**     `resnet50`                          25.56       4.12      123.91          8.1       0.8628

  **HydraSwin (Đề     `swin_tiny_patch4_window7`          28.29       4.50      126.10          7.9   **0.9600**
  xuất)**                                                                                           

  **ViT-Tiny**        `vit_tiny_patch16_224`               5.72       1.26       48.50         20.6       0.9411
  --------------------------------------------------------------------------------------------------------------

### Phát hiện chính và sự đánh đổi

1.  **Triển khai trên thiết bị biên (Edge Deployment):** Trong benchmark
    này, MobileNetV3 đạt hiệu suất cao nhất về tốc độ, với latency 24.13
    ms và 41.4 FPS, trong khi chỉ có 5.48M parameters.
2.  **Ngữ cảnh toàn cục (Global Context):** HydraSwin sử dụng cơ chế
    attention cửa sổ dịch chuyển phân cấp để mô hình hóa các mối quan hệ
    không cục bộ giữa những thuộc tính sản phẩm.
3.  **Hiệu chỉnh Neuro-Symbolic:** Bộ lọc luật được kích hoạt trên 148
    mẫu kiểm thử và đạt độ chính xác 91.89%. Bộ lọc cải thiện Season
    Macro F1 từ 0.8910 lên 0.9852.

------------------------------------------------------------------------

## 📂 Cấu trúc Repository

``` text
HydraSwin-VNICT2026/
├── models/
│   ├── hydraswin.py          # Kiến trúc HydraSwin đa nhánh (Swin-Tiny)
│   └── baselines.py          # Các mô hình baseline MobileNetV3, ConvNeXt-Tiny, ViT
├── loss/
│   └── weighted_loss.py      # Hàm loss có trọng số theo Inverse-Frequency
├── postprocess/
│   └── rule_filter.py        # Bộ lọc nhất quán Neuro-Symbolic gồm 42 luật
├── notebooks/
│   └── hydraswin_training_pipeline.ipynb # Notebook huấn luyện và đánh giá gốc
├── benchmark_results/
│   ├── benchmark_efficiency_results.csv
│   └── rule_ablation_results.csv
├── requirements.txt          # Các thư viện Python cần thiết
├── LICENSE                   # Giấy phép MIT
└── README.md
```

------------------------------------------------------------------------

## 📖 Trích dẫn

Nếu sử dụng mã nguồn hoặc kết quả nghiên cứu này trong công trình của
mình, vui lòng trích dẫn bài báo VNICT 2026:

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

## 👨‍💻 Tác giả

-   **Dang D. Nguyen** --- Viện Công nghệ Thông minh và Tương tác, Đại
    học Kinh tế Thành phố Hồ Chí Minh (UEH).\
    **Email:** <nguyendidang1207@gmail.com> \| **ORCID:**
    [0009-0003-1971-8258](https://orcid.org/0009-0003-1971-8258)
-   **Bao T. Nguyen** --- Đại học Kinh tế Thành phố Hồ Chí Minh (UEH).
-   **Hoc T. Duong** --- Đại học Kinh tế Thành phố Hồ Chí Minh (UEH).

------------------------------------------------------------------------

## 📄 Giấy phép

Dự án được phát hành theo [Giấy phép MIT](LICENSE).