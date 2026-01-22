# MTA-CLIP (Multi-Task Adaptive CLIP for Person Re-Identification)

This repository provides an implementation of **MTA-CLIP**, proposed in our paper:

> **Multi-Task Adaptive CLIP for Enhanced Person Re-Identification Across Domains (MTA-CLIP)**

MTA-CLIP adapts a pretrained CLIP backbone to person re-identification with:

- **HCP** (Hierarchical Context Prompting) for shared + domain-specific semantics, and
- **SGA** (Semantic Grouping Adapter), a lightweight visual adaptation module for fine-grained identity cues.

---

## SGA (Semantic Grouping Adapter)

SGA is plugged into the visual branch (ViT-based image encoder) during fine-tuning.

Given the patch-token feature map from the last Transformer layer, SGA:

- splits channels into multiple semantic groups,
- performs group-wise lightweight transformations / reweighting,
- integrates a global context branch,
- fuses the two streams via learnable softmax weights,
- outputs an enhanced feature map through a residual connection.

---

## Training strategy: Progressive Two-Stage Optimization (PTSO)

### Stage 1: Semantic alignment (prompt learning only)

- Freeze **image encoder** and **text encoder**.
- **SGA is not trained / not activated**.
- Optimize **HCP** to decouple
  - shared prompts $\mathbf{P}_S$
  - domain-specific prompts $\mathbf{P}_{D_k}$
- Use a cross-modal contrastive objective (coarse-grained alignment):

$$
\mathcal{L}_{stage1} = \sum_{k=1}^{K} \mathcal{L}_{con}(\mathcal{D}_k)
$$

### Stage 2: Multi-task fine-tuning (enable SGA)

- Initialize from Stage 1.
- Enable **SGA** and partially unfreeze the visual branch.
- Jointly optimize identity discrimination + cross-modal consistency:

$$
\mathcal{L}_{total}(\mathcal{D}_k) =
\mathcal{L}_{id}(\mathcal{D}_k)
+ \lambda_1 \mathcal{L}_{tri}(\mathcal{D}_k)
+ \lambda_2 \mathcal{L}_{i2t}(\mathcal{D}_k)
$$

> Note: In our diagrams/captions, Stage 1 may use $\mathcal{L}_{con}$ (symmetric), while Stage 2 highlights $\mathcal{L}_{i2t}$ to emphasize the image-to-text direction used for maintaining alignment during fine-tuning.

---

## Where is SGA in the code?

SGA is implemented and used in the model code under:

- `model/` (visual backbone / adapters)

Training scripts (naming may vary by experiment setting):

- `train_clipreid_afem_multitask.py` (multi-task training)
- `train_clipreid_afem.py` (single setting)
- `train_clipreid.py` (baseline / other configs)

Visualization utilities used in the paper:

- Extract real features (cached to `./fig/features/`):
  - `extract_real_features.py`
- t-SNE method comparison (Baseline vs Ours):
  - `draw_method_comparison.py`

---

## Expected effect

Qualitatively (t-SNE / activation maps):

- more compact intra-class clusters
- clearer inter-class separation
- stronger activation on identity-relevant regions

---

## Acknowledgements

This codebase is built upon CLIP-ReID and related open-source ReID toolkits. If you use this repository, please cite the original works as well as our paper.
