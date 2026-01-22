# SGA (Semantic Grouping Adapter)

This repository extends the CLIP-ReID codebase with **SGA (Semantic Grouping Adapter)**, a lightweight visual adaptation module introduced in our paper:

> **Multi-Task Adaptive CLIP for Enhanced Person Re-Identification Across Domains (MTA-CLIP)**

SGA is designed to **enhance fine-grained identity-discriminative cues** (e.g., clothing texture, carried items, subtle appearance differences) while keeping the overall CLIP representation stable and efficient.

---

## 1. What is SGA?

**SGA (Semantic Grouping Adapter)** is a plug-in module inserted into the visual branch (ViT-based image encoder) during fine-tuning.

Given the patch-token feature map from the last Transformer layer, SGA:

- **splits channels into multiple semantic groups**,
- performs **group-wise lightweight transformations / reweighting**,
- integrates a **global context branch** to preserve consistency,
- fuses the two streams via **learnable softmax weights**,
- outputs an enhanced feature map through a **residual connection**.

This yields stronger local semantic awareness with negligible computational overhead.

---

## 2. How SGA is used in training (PTSO)

Our training follows a **Progressive Two-Stage Optimization (PTSO)** strategy:

### Stage 1: Semantic alignment (prompt learning only)

- Freeze **image encoder** and **text encoder**.
- **SGA is not trained / not activated**.
- Optimize **Hierarchical Context Prompting (HCP)** to decouple:
  - shared prompts \(\mathbf{P}_S\)
  - domain-specific prompts \(\mathbf{P}_{D_k}\)
- Use a cross-modal contrastive objective (coarse-grained alignment):

\[
\mathcal{L}_{stage1} = \sum_{k=1}^{K} \mathcal{L}_{con}(\mathcal{D}_k)
\]

### Stage 2: Multi-task fine-tuning (enable SGA)

- Initialize from Stage 1.
- Enable **SGA** and partially unfreeze the visual branch.
- Jointly optimize identity discrimination + cross-modal consistency:

\[
\mathcal{L}_{total}(\mathcal{D}_k) =
\mathcal{L}_{id}(\mathcal{D}_k)
+ \lambda_1 \mathcal{L}_{tri}(\mathcal{D}_k)
+ \lambda_2 \mathcal{L}_{i2t}(\mathcal{D}_k)
\]

> Note: In our diagrams/captions, Stage 1 sometimes uses \(\mathcal{L}_{con}\) (symmetric), while Stage 2 highlights \(\mathcal{L}_{i2t}\) to emphasize the image-to-text direction used for maintaining alignment during fine-tuning.

---

## 3. Where is SGA in the code?

SGA is implemented and used in the model code under:

- `model/` (visual backbone / adapters)
- training scripts:
  - `train_clipreid_sga_multitask.py`
  - `train_clipreid_sga.py`
  - `train_clipreid.py`

If you are looking for **visualization utilities** used in the paper:

- Extract real features (cached to `./fig/features/`):
  - `extract_real_features.py`
- t-SNE method comparison (Baseline vs Ours):
  - `draw_method_comparison.py`

---

## 4. Expected effect

Qualitatively (t-SNE / activation maps):

- more compact intra-class clusters
- clearer inter-class separation
- stronger activation on identity-relevant regions

Quantitatively (examples from the paper draft):

- Market-1501: improves mAP / Rank-1 over baseline
- MSMT17: improves robustness under domain shifts

---

## 5. Acknowledgements

This codebase is built upon CLIP-ReID and related open-source ReID toolkits. If you use this repository, please cite the original works as well as our paper.
# MTA-CLIP
