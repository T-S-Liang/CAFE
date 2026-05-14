<div align="center">

<img src="https://raw.githubusercontent.com/T-S-Liang/CAFE/gh-pages/static/images/cafe_logo.png" width="180">

# CAFE: From Pixels to Concepts — Do Segmentation Models Understand What They Segment?

[![Page](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://t-s-liang.github.io/CAFE)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.09591)
[![PDF](https://img.shields.io/badge/arXiv-PDF-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2605.09591)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-CAFE-blue)](https://huggingface.co/datasets/teemosliang/CAFE)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[Shuang Liang](https://t-s-liang.github.io)<sup>1,3†</sup>,
Zeqing Wang<sup>2†</sup>,
Yuxian Li<sup>1†</sup>,
Xihui Liu<sup>1</sup>,
Han Wang<sup>1,3*</sup>

<sup>1</sup>The University of Hong Kong, <sup>2</sup>Sun Yat-sen University, <sup>3</sup>CASIC, HKU

<sup>†</sup>Equal contribution. <sup>*</sup>Corresponding author.

</div>

---

## 📢 News

- **[2026-May-14]** 📄 Paper released on [arXiv](https://arxiv.org/abs/2605.09591).
- **[2026-May-09]** 🤖 **CAFE-SAM3 agent** code released under [`agent/`](agent/).
- **[2026-May-09]** 🧪 **CAFEval2026 evaluation toolkit** released under [`tools/`](tools/) (cgF<sub>1</sub> + AFPR/UFPR/IL-FPR/ACSR/UCSR/CSR/SoftSwap).
- **[2026-May-09]** 🤗 **CAFEval2026 benchmark** (2,146 paired samples) released on [HuggingFace](https://huggingface.co/datasets/teemosliang/CAFE).
- **[2026-May-09]** 🌐 [Project page](https://t-s-liang.github.io/CAFE) with interactive leaderboard live.

### ✅ Released

- [x] **CAFE-SAM3 agent** release
- [x] **CAFEval2026 evaluation toolkit** release
- [x] **CAFEval2026 benchmark** release
- [x] **Project page** with interactive leaderboard

---

## 🔥 Highlights

**CAFE** asks a sharper question than *is the mask accurate?*: **does the model actually ground the queried concept, or is it riding on visually salient but semantically misleading cues?**

- ✅ **Attribute-level counterfactual evaluation**: target region and ground-truth mask are fixed; only attributes (appearance, context, material) change.
- ✅ **Paired positive vs. misleading-negative prompts** expose the gap between mask quality and concept discrimination.
- ✅ **Three failure regimes**: Superficial Mimicry, Context Conflict, Ontological Conflict — 2,146 paired samples in total.
- ✅ **CAFE-SAM3 agent**: an MLLM-driven loop that corrects SAM3 failures via concept verification, zoom-in inspection, and re-prompting.

---

## 🎨 Visualization

### Task overview

<div align="center">
<img src="https://raw.githubusercontent.com/T-S-Liang/CAFE/gh-pages/static/images/teaser.png" width="100%">
</div>

### A slice of the benchmark

<div align="center">
<img src="https://raw.githubusercontent.com/T-S-Liang/CAFE/gh-pages/static/images/more_cases.jpg" width="100%">
</div>

---

## 🧪 Three Counterfactual Settings

| Split | # samples | What changes | Stress test |
| --- | :---: | --- | --- |
| **Superficial Mimicry (SM)** | 1,111 | Surface texture / appearance | Pattern vs. object identity |
| **Context Conflict (CC)** | 593 | Surrounding environment | Scene prior vs. true category |
| **Ontological Conflict (OC)** | 442 | Material / substance | Shape shortcut vs. concept |

The mask stays put; the attribute moves. A faithful model should flip its prediction between the positive and the misleading-negative prompt, *not* ride the leftover visual cue.

---

## 📊 Leaderboard (cgF<sub>1</sub> ↑)

`cgF₁` is the headline metric, combining concept discrimination (`IL_MCC`) with mask quality (`pmF₁`).

| # | Model | Type | SM | CC | OC | **Overall** |
| :---: | --- | --- | :---: | :---: | :---: | :---: |
| 1 | **CAFE-SAM3 (GPT-5.5)** | Agentic | 69.7 | 66.1 | 44.7 | **63.3** |
| 2 | SAM 3 | End-to-end | 53.0 | 61.4 | -10.5 | 38.5 |
| 3 | OWLv2 + SAM 1 | Multi-model | 43.2 | 41.0 | -8.0 | 27.9 |
| 4 | YOLO-World | End-to-end | 39.4 | 20.8 | -5.9 | 21.1 |
| 5 | OpenSeeD | End-to-end | 28.9 | 29.8 | -4.0 | 15.1 |
| 6 | Grounded SAM 2 | Multi-model | 13.0 | 5.9 | 3.6 | 9.9 |

> 💡 Strong mask predictors are **not** automatically strong concept grounders. SAM 3 collapses on OC (negative cgF<sub>1</sub>); the CAFE-SAM3 agent recovers most of the loss through reasoning. See the [interactive leaderboard](https://t-s-liang.github.io/CAFE/#Leaderboard) for `IL_MCC`, `pmF₁`, `FPR`, `AFPR`, `ACSR`.

---

## 🛠️ Setup

### Installation

```bash
git clone https://github.com/T-S-Liang/CAFE.git
cd CAFE

conda create -n cafe python=3.10 -y
conda activate cafe

pip install numpy scipy tqdm pycocotools requests
```

### SAM 3 (required for `cgF₁` evaluation and the CAFE-SAM3 agent)

`tools/cgf1_eval_wrapper.py` imports SAM 3's official `CGF1Evaluator`, and the agent uses SAM 3 as the segmentation backbone. Clone SAM 3 alongside CAFE and point `--sam3-dir` (or the `SAM3_DIR` env var) at its repository root:

```bash
git clone https://github.com/facebookresearch/sam3.git
export SAM3_DIR=$PWD/sam3
```

### Dataset

```bash
huggingface-cli download teemosliang/CAFE --repo-type dataset --local-dir CAFEval2026
```

Expected layout used by `tools/run_eval.sh` defaults:

```
CAFEval2026/
├── CAFEval2026_annotations.json   # full set (SM + CC + OC)
├── CAFEval2026_SM.json            # Superficial Mimicry subset
├── CAFEval2026_CC.json            # Context Conflict subset
├── CAFEval2026_OC.json            # Ontological Conflict subset
└── CAFEval2026_imgs/              # image directory
```

---

## 🕹️ Evaluation

CAFEval2026 evaluates any **COCO-format segmentation prediction dump** (`[{image_id, category_id, segmentation: RLE, score, ...}, ...]`). One command produces both the SAM 3 `cgF₁` family (per-subset) and the CAFE custom metrics (AFPR, UFPR, IL-FPR, ACSR, UCSR, CSR, SoftSwap).

### One-click evaluation

```bash
bash tools/run_eval.sh \
    --pred   path/to/coco_predictions_segm.json \
    --out-dir results/<your_run_tag> \
    --gt-ann CAFEval2026/CAFEval2026_annotations.json \
    --gt-sm  CAFEval2026/CAFEval2026_SM.json \
    --gt-cc  CAFEval2026/CAFEval2026_CC.json \
    --gt-oc  CAFEval2026/CAFEval2026_OC.json \
    --img-dir CAFEval2026/CAFEval2026_imgs \
    --sam3-dir $SAM3_DIR
```

Outputs written to `--out-dir`:

| File | Contents |
| --- | --- |
| `<TAG>__cgf1_t<T>_{Overall,SM,CC,OC}.log` | SAM 3 cgF<sub>1</sub> stdout per subset |
| `<TAG>__cafe_t<T>_tau<TAU>.json` | CAFE custom metrics: per-subset summary + per-pair audit |
| `<TAG>__SUMMARY_t<T>_tau<TAU>.md` | Consolidated markdown report (cgF<sub>1</sub> + CAFE) |

Tunable knobs:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--score-thr` | `0.5` (SAM 3 paper) | Presence-confidence threshold; preds below this are dropped |
| `--iou-thr` | `0.3` (CAFE paper) | Target-alignment IoU threshold (CAFE custom metrics only; cgF<sub>1</sub> sweeps IoU 0.50:0.95) |
| `--sam3-dir` | `$SAM3_DIR` | SAM 3 repository root, must contain `sam3/eval/cgf1_eval.py` |
| `--python` | `python3` | Python interpreter |

### Evaluate any model

Drop your model's predictions into COCO segm format and rerun the same script — the toolkit is model-agnostic:

```python
[
  {"image_id": 12345, "category_id": 1, "score": 0.87,
   "segmentation": {"size": [H, W], "counts": "<rle>"}},
  ...
]
```

### Run only the CAFE custom metrics

If you already have `cgF₁` numbers and want just AFPR / ACSR / SoftSwap / etc.:

```bash
python tools/eval_cafe_metrics.py \
    --ann  CAFEval2026/CAFEval2026_annotations.json \
    --pred path/to/coco_predictions_segm.json \
    --score-threshold 0.5 \
    --iou-threshold   0.3 \
    --out  results/<tag>/cafe_metrics.json
```

---

## 🤖 CAFE-SAM3 Agent

The CAFE-SAM3 agent wraps SAM 3 with an MLLM-driven loop that performs concept verification, zoom-in inspection, and re-prompting on hard counterfactual cases. Lives in [`agent/`](agent/).

### Run the agent

```bash
export OPENAI_API_KEY=...
export CAFE_ANN=CAFEval2026/CAFEval2026_annotations.json
export CAFE_IMG_DIR=CAFEval2026/CAFEval2026_imgs

python agent/eval_agent.py \
    --provider ttapi_openai \
    --model    gpt-5.5 \
    --output-dir agent_eval_output \
    --concurrency 4 \
    --max-turns 10
```

Other modes:

```bash
python agent/eval_agent.py --model gpt-5.5 --force         # re-run all (ignore checkpoint)
```

The runner is **resumable**: completed cases are written to `agent_eval_output/checkpoint.json` and skipped on restart. Predictions are produced in the same COCO segm format consumed by `tools/run_eval.sh`, so you can score the agent the same way as any baseline:

```bash
bash tools/run_eval.sh \
    --pred   agent_eval_output/coco_predictions_segm.json \
    --out-dir results/cafe_sam3_gpt55
```

### Live demo

A click-through of the agent's reasoning trace (including cases where SAM 3 alone fails) is on the project page:

🌐 **[Agent demo](https://t-s-liang.github.io/CAFE/#AgentDemo)**

---

## 🎓 Citation

If you find CAFE useful in your research, please cite:

```bibtex
@article{liang2026pixels,
  title={From Pixels to Concepts: Do Segmentation Models Understand What They Segment?},
  author={Liang, Shuang and Wang, Zeqing and Li, Yuxian and Liu, Xihui and Wang, Han},
  journal={arXiv preprint arXiv:2605.09591},
  year={2026}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

CAFE is built on top of, and grateful to, the following open-source efforts:

- [SAM 3](https://github.com/facebookresearch/sam3) and [SAM 2](https://github.com/facebookresearch/sam2): Promptable segmentation backbones, evaluated as baselines and used inside the CAFE-SAM3 agent.
- [OWLv2](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [OpenSeeD](https://github.com/IDEA-Research/OpenSeeD), [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2): Open-vocabulary baselines included in the leaderboard.
- [LVIS](https://www.lvisdataset.org/) and [COCO](https://cocodataset.org/): Source images and category vocabulary used for counterfactual construction.

---

## 📧 Contact

For questions, suggestions, or collaboration inquiries:

- **Shuang Liang**: [sliang57@connect.hku.hk](mailto:sliang57@connect.hku.hk)
- **Project page**: [https://t-s-liang.github.io/CAFE](https://t-s-liang.github.io/CAFE)

---

<div align="center">

**⭐ Star us on GitHub if CAFE helps your work — it motivates us a lot!**

[🌐 Website](https://t-s-liang.github.io/CAFE) | [📄 Paper](https://arxiv.org/abs/2605.09591) | [🤗 Dataset](https://huggingface.co/datasets/teemosliang/CAFE) | [💻 Code](https://github.com/T-S-Liang/CAFE)

</div>
