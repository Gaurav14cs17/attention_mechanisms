<div align="center">

<!-- HERO SECTION -->
<img src="https://img.shields.io/badge/📚_Surveys-3-blue?style=for-the-badge" alt="Surveys"/>
<img src="https://img.shields.io/badge/🎨_Diagrams-55+-purple?style=for-the-badge" alt="Diagrams"/>
<img src="https://img.shields.io/badge/❓_Q&A-130+-green?style=for-the-badge" alt="Q&A"/>

# ⚡ Attention Mechanisms
### The Complete Visual Encyclopedia

---

```
████████╗██████╗  █████╗ ███╗   ██╗███████╗███████╗ ██████╗ ██████╗ ███╗   ███╗███████╗██████╗ 
╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔════╝██╔══██╗
   ██║   ██████╔╝███████║██╔██╗ ██║███████╗█████╗  ██║   ██║██████╔╝██╔████╔██║█████╗  ██████╔╝
   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║██╔══╝  ██╔══██╗
   ██║   ██║  ██║██║  ██║██║ ╚████║███████║██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗██║  ██║
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
                              ███████╗███████╗███████╗██╗ ██████╗██╗███████╗███╗   ██╗ ██████╗██╗   ██╗
                              ██╔════╝██╔════╝██╔════╝██║██╔════╝██║██╔════╝████╗  ██║██╔════╝╚██╗ ██╔╝
                              █████╗  █████╗  █████╗  ██║██║     ██║█████╗  ██╔██╗ ██║██║      ╚████╔╝ 
                              ██╔══╝  ██╔══╝  ██╔══╝  ██║██║     ██║██╔══╝  ██║╚██╗██║██║       ╚██╔╝  
                              ███████╗██║     ██║     ██║╚██████╗██║███████╗██║ ╚████║╚██████╗   ██║   
                              ╚══════╝╚═╝     ╚═╝     ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝   ╚═╝   
```

*Three in-depth surveys covering everything you need to know about efficient transformer architectures*

**From O(n²) → O(n) | FlashAttention → Mamba | ViT → DINOv2**

[📖 Efficient Attention](#-efficient-attention-survey) · [⚡ Faster Transformers](#-faster-and-lighter-transformers) · [👁️ Vision Transformers](#️-vision-transformers)

</div>

---

## 🎯 What's Inside?

<table>
<tr>
<td width="33%" align="center">

### 📖 Efficient Attention

**19 Diagrams**

FlashAttention • GQA/MQA  
Sparse Patterns • Mamba  

</td>
<td width="33%" align="center">

### ⚡ Faster & Lighter

**29 Diagrams**

Quantization • Pruning  
Distillation • MoE  

</td>
<td width="33%" align="center">

### 👁️ Vision Transformers

**16 Diagrams**

ViT • Swin • MAE  
DINO • 130+ Q&A  

</td>
</tr>
</table>

---

<div align="center">

## 📚 Survey Collection

</div>

---

## 📖 Efficient Attention Survey

<table>
<tr>
<td width="70px" align="center">
<h1>🔥</h1>
</td>
<td>

### Hardware-efficient, Sparse, Compact, and Linear Attention

**The definitive guide to making attention O(n) without sacrificing quality**

</td>
<td width="120px" align="center">

[![Read](https://img.shields.io/badge/Read-Blog-blue?style=for-the-badge)](./Efficient_Attention_Survey/BLOG_README.md)

</td>
</tr>
</table>

<table>
<tr>
<td width="50%">

#### 🏗️ Four Pillars of Efficient Attention

| Class | Core Idea | Key Methods |
|:-----:|:----------|:------------|
| ⚡ | **Hardware-efficient** | FlashAttention, SageAttention |
| 📦 | **Compact** | MQA, GQA, MLA (KV compression) |
| 🎯 | **Sparse** | Longformer, BigBird, H2O |
| 🔄 | **Linear** | Mamba, RWKV, RetNet |

</td>
<td width="50%">

#### 📋 You'll Learn

```
✓ GPU memory hierarchy (HBM vs SRAM)
✓ Prefilling vs Decoding optimization
✓ KV cache compression techniques
✓ Gating mechanisms in linear attention
✓ Test-Time Training (TTT)
```

</td>
</tr>
</table>

<details>
<summary><b>📊 View All 19 Visualizations</b></summary>

| | Diagram | Description |
|:-:|:--------|:------------|
| 1 | `overview_attention_types.svg` | Four classes of efficient attention |
| 2 | `standard_attention_explained.svg` | Step-by-step attention mechanism |
| 3 | `gpu_memory_hierarchy.svg` | HBM vs SRAM explained |
| 4 | `flash_attention.svg` | FlashAttention tiling strategy |
| 5 | `compact_attention.svg` | MQA, GQA, MLA comparison |
| 6 | `sparse_attention.svg` | Sparse attention patterns |
| 7 | `linear_attention_forms.svg` | Parallel, Recurrent, Chunkwise |
| 8 | `linear_attention_methods.svg` | Mamba, RWKV, RetNet |
| 9 | `gating_mechanisms.svg` | Forget and select gates |
| 10 | `test_time_training.svg` | TTT approach |
| 11-19 | `formula_*.svg` | Mathematical formulations |

</details>

---

## ⚡ Faster and Lighter Transformers

<table>
<tr>
<td width="70px" align="center">
<h1>🚀</h1>
</td>
<td>

### A Practical Survey on Making Transformers Deployable

**Distillation → Pruning → Quantization = 40-85× compression**

</td>
<td width="120px" align="center">

[![Read](https://img.shields.io/badge/Read-Blog-green?style=for-the-badge)](./Faster%20and%20Lighter%20Transformers/README.md)

</td>
</tr>
</table>

<table>
<tr>
<td width="50%">

#### 🛠️ Complete Efficiency Toolkit

| Method | What It Does | Savings |
|:------:|:-------------|:-------:|
| 📚 | **Knowledge Distillation** | 5-7× smaller |
| 🔢 | **Quantization** | 4-32× smaller |
| ✂️ | **Pruning** | 2-4× smaller |
| 🔗 | **Weight Sharing** | 12-18× smaller |
| 🔀 | **MoE** | 5× faster training |
| 💾 | **Checkpointing** | 10× less memory |

</td>
<td width="50%">

#### 📋 You'll Learn

```
✓ DistilBERT, TinyBERT, MobileBERT
✓ INT8, Mixed Precision, QAT
✓ Structured vs Unstructured Pruning
✓ ALBERT's 18× parameter reduction
✓ Switch Transformer & trillion-scale
✓ Combining techniques effectively
```

</td>
</tr>
</table>

<details>
<summary><b>📊 View All 29 Visualizations</b></summary>

| | Diagram | Description |
|:-:|:--------|:------------|
| 1-5 | Architecture | Overview, Transformer, RNN comparison |
| 6-10 | Formulas | Attention equations explained |
| 11-15 | Distillation | Teacher-student, KD loss |
| 16-20 | Compression | Quantization, Pruning methods |
| 21-25 | Efficiency | Weight sharing, MoE, GPipe |
| 26-29 | Comparison | Model sizes, complexity tables |

</details>

---

## 👁️ Vision Transformers

<table>
<tr>
<td width="70px" align="center">
<h1>👁️</h1>
</td>
<td>

### The Complete Visual Guide to Vision Transformers

**ViT → DeiT → Swin → MAE → DINOv2 + 130 Interview Questions**

</td>
<td width="120px" align="center">

[![Read](https://img.shields.io/badge/Read-Blog-purple?style=for-the-badge)](./Vision_Transformers/README.md)

</td>
</tr>
</table>

<table>
<tr>
<td width="50%">

#### 🗺️ Complete ViT Landscape

| Category | Key Models |
|:--------:|:-----------|
| 🔷 | **Core ViT** — ViT, DeiT, ViT-H/G |
| 🏗️ | **Hierarchical** — Swin, CSWin, Focal |
| 🔀 | **Hybrid** — CvT, ConViT, CoAtNet, LeViT |
| 📱 | **Mobile** — PVT, MobileViT, EdgeViT |
| 🎓 | **Self-Supervised** — MAE, BEiT, DINO |
| 🎯 | **Tasks** — DETR, Mask2Former |

</td>
<td width="50%">

#### 📋 You'll Learn

```
✓ Patch embedding and [CLS] token
✓ Window-based attention (Swin)
✓ Mobile deployment strategies
✓ Self-supervised pre-training
✓ Object detection with DETR
✓ 130+ interview Q&A with diagrams
```

</td>
</tr>
</table>

<details>
<summary><b>📊 View All 16 Visualizations</b></summary>

**Main Diagrams (7):**
| | Diagram | Description |
|:-:|:--------|:------------|
| 1 | `vit_taxonomy.svg` | Complete ViT taxonomy |
| 2 | `core_vit_architecture.svg` | ViT architecture |
| 3 | `swin_transformer.svg` | Hierarchical Swin |
| 4 | `hybrid_cnn_transformer.svg` | CNN + Transformer |
| 5 | `efficient_mobile_vit.svg` | Mobile variants |
| 6 | `self_supervised_vit.svg` | MAE, DINO |
| 7 | `vision_tasks.svg` | Detection, Segmentation |

**Interview Q&A Diagrams (9):**
| | Diagram | Topic |
|:-:|:--------|:------|
| 1 | `patch_embedding_process.svg` | How patches work |
| 2 | `cls_token_explained.svg` | [CLS] token role |
| 3 | `vit_vs_cnn_comparison.svg` | ViT vs CNN |
| 4 | `position_encoding_types.svg` | Position encodings |
| 5 | `attention_complexity.svg` | O(n²) explained |
| 6 | `window_vs_global_attention.svg` | Local vs global |
| 7 | `multi_head_attention.svg` | Multi-head mechanism |
| 8 | `mae_pretraining.svg` | MAE approach |
| 9 | `deit_distillation.svg` | DeiT training |

</details>

---

<div align="center">

## 🧭 Quick Navigation

</div>

### 🔥 Hot Topics

| Topic | Survey | Quick Link |
|:------|:------:|:-----------|
| **FlashAttention** | Efficient Attention | [→ Hardware-efficient](./Efficient_Attention_Survey/BLOG_README.md#-class-1-hardware-efficient-attention) |
| **KV Cache (GQA/MQA)** | Efficient Attention | [→ Compact Attention](./Efficient_Attention_Survey/BLOG_README.md#-class-2-compact-attention) |
| **Mamba/RWKV** | Efficient Attention | [→ Linear Attention](./Efficient_Attention_Survey/BLOG_README.md#-class-4-linear-attention) |
| **Quantization (INT8)** | Faster & Lighter | [→ Quantization](./Faster%20and%20Lighter%20Transformers/README.md#quantization) |
| **DistilBERT** | Faster & Lighter | [→ Distillation](./Faster%20and%20Lighter%20Transformers/README.md#knowledge-distillation) |
| **Swin Transformer** | Vision Transformers | [→ Hierarchical ViT](./Vision_Transformers/README.md#-hierarchical-vision-transformers) |
| **MAE / DINO** | Vision Transformers | [→ Self-Supervised](./Vision_Transformers/README.md#-self-supervised-vision-transformers) |
| **Interview Prep** | Vision Transformers | [→ 130+ Q&A](./Vision_Transformers/interview_qa/README.md) |

---

<div align="center">

## 📊 Method Comparison

</div>

### ⚡ Efficiency Trade-offs

| Approach | Speed | Memory | Quality | Effort |
|:---------|:-----:|:------:|:-------:|:------:|
| **FlashAttention** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Drop-in |
| **GQA/MQA** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Retrain |
| **Sparse Attention** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Retrain |
| **Linear Attention** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Full train |
| **Quantization** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | PTQ/QAT |
| **Pruning** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Fine-tune |
| **Distillation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Full train |
| **MoE** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Full train |

### 🎯 When to Use What?

```
┌─────────────────────────────────────────────────────────────────────┐
│  DECISION TREE: Which efficiency method should I use?               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ❓ Need to handle long sequences (>2K tokens)?                     │
│     ├── YES → FlashAttention (drop-in) or Sparse/Linear attention  │
│     └── NO  → Standard attention is fine                           │
│                                                                     │
│  ❓ Need smaller model for deployment?                              │
│     ├── YES → Distillation → Pruning → Quantization (combine!)     │
│     └── NO  → Keep original model                                  │
│                                                                     │
│  ❓ Limited memory during training?                                 │
│     ├── YES → Gradient checkpointing + Mixed precision             │
│     └── NO  → Standard training                                    │
│                                                                     │
│  ❓ Working with images?                                            │
│     ├── YES → Swin (dense) or ViT (classification)                 │
│     └── NO  → Text transformers                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

<div align="center">

## 📁 Repository Structure

</div>

```
attention_mechanisms/
│
├── 📖 README.md                              ← You are here
│
├── 📂 Efficient_Attention_Survey/
│   ├── BLOG_README.md                        ← Main blog post
│   ├── README.md                             ← Technical summary
│   ├── svg_figs/                             ← 19 diagrams
│   ├── png_figs/                             ← Original paper figures
│   └── resource/                             ← Source PDFs
│
├── 📂 Faster and Lighter Transformers/
│   ├── README.md                             ← Main blog post
│   ├── svg_figs/                             ← 29 diagrams
│   └── *.pdf                                 ← Source paper
│
└── 📂 Vision_Transformers/
    ├── README.md                             ← Main blog post
    ├── svg_figs/                             ← 7 diagrams
    └── interview_qa/
        ├── README.md                         ← 130+ Q&A
        └── svg_figs/                         ← 9 explanatory diagrams
```

---

<div align="center">

## 🚀 Getting Started

</div>

### 📚 Learning Path

```
                    ┌─────────────────────────────────────┐
                    │        NEW TO TRANSFORMERS?         │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────┐
                    │  1. Understand the O(n²) Problem    │
                    │     → Efficient Attention Survey    │
                    └───────────────┬─────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
┌─────────▼─────────┐   ┌───────────▼───────────┐   ┌─────────▼─────────┐
│   TEXT/LLMs?      │   │    DEPLOYMENT?        │   │     VISION?       │
│                   │   │                       │   │                   │
│ Linear Attention  │   │ Distillation +        │   │ ViT → Swin →      │
│ FlashAttention    │   │ Quantization          │   │ Mobile ViT        │
│ GQA/MQA           │   │ Pruning               │   │ MAE/DINO          │
└───────────────────┘   └───────────────────────┘   └───────────────────┘
```

### 📖 Recommended Reading Order

| Step | What | Link |
|:----:|:-----|:-----|
| 1️⃣ | Understand the quadratic bottleneck | [Efficient Attention → Problem](./Efficient_Attention_Survey/BLOG_README.md#-the-quadratic-bottleneck-problem) |
| 2️⃣ | Learn standard attention step-by-step | [Efficient Attention → Standard](./Efficient_Attention_Survey/BLOG_README.md#-standard-attention-step-by-step) |
| 3️⃣ | Explore all efficiency methods | [Faster & Lighter → Taxonomy](./Faster%20and%20Lighter%20Transformers/README.md#taxonomy-of-efficiency-methods) |
| 4️⃣ | Vision Transformer fundamentals | [Vision → Core ViT](./Vision_Transformers/README.md#-core-vit-architecture) |
| 5️⃣ | Test your knowledge | [Interview Q&A](./Vision_Transformers/interview_qa/README.md) |

---

<div align="center">

## 📄 Source Papers

</div>

| Survey | Paper | Authors | Year |
|:-------|:------|:--------|:----:|
| Efficient Attention | [Attention Survey](https://attention-survey.github.io) | Zhang et al. | 2025 |
| Faster & Lighter | [ACM Computing Surveys](./Faster%20and%20Lighter%20Transformers/A%20Practical%20Survey%20on%20Faster%20and%20Lighter%20Transformers.pdf) | Fournier et al. | 2023 |

---


