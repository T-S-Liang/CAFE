# CAFE: Counterfactual Attribute Factuality Evaluation

**From Pixels to Concepts: Do Segmentation Models Understand What They Segment?**

Shuang Liang, Zeqing Wang, Yuxian Li, Xihui Liu, Han Wang

[[Paper]](https://arxiv.org/abs/2605.09591) [[Project Page]](https://t-s-liang.github.io/CAFE/) [[Dataset]](https://huggingface.co/datasets/teemosliang/CAFE) [[Code]](https://github.com/T-S-Liang/CAFE)

CAFE is a benchmark for evaluating concept-faithful segmentation in promptable segmentation models. It tests whether models faithfully ground queried concepts or rely on visually salient but semantically misleading cues, using 2,146 paired counterfactual samples across three edit types:

- **Superficial Mimicry (SM)**: surface texture changed to mimic another category
- **Context Conflict (CC)**: object placed in misleading context
- **Ontological Conflict (OC)**: material/substance changed while shape preserved
