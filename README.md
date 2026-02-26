# STAR-LDM: Stop-Think-AutoRegress Language Diffusion Model

**[Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning](https://openreview.net/forum?id=c05qIG1Z2B)**

Justin Lovelace, Christian Belardi, Sofian Zalouk, Adhitya Polavaram, Srivatsa Kundurthy, Kilian Q Weinberger

_Conference on Language Modeling (COLM) 2025_

---

![STAR-LDM Overview](fig/starldm.png)

## Abstract

The Stop-Think-AutoRegress Language Diffusion Model (STAR-LDM) integrates latent diffusion planning with autoregressive generation. Unlike conventional autoregressive language models limited to token-by-token decisions, STAR-LDM incorporates a "thinking" phase that pauses generation to refine a semantic plan through diffusion before continuing. This enables global planning in continuous space prior to committing to discrete tokens. Evaluations show STAR-LDM significantly outperforms similar-sized models on language understanding benchmarks and achieves >70% win rates in LLM-as-judge comparisons for narrative coherence and commonsense reasoning. The architecture also allows straightforward control through lightweight classifiers, enabling fine-grained steering of attributes without model retraining while maintaining better fluency-control trade-offs than specialized approaches.

## Setup

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU (tested on A6000/H100)

### Installation

```bash
# Clone the repository
git clone https://github.com/<org>/STAR-LDM.git
cd STAR-LDM

# Install dependencies (inference)
pip install -r requirements.txt

# Additional dependencies for training
pip install accelerate>=1.0 wandb datasets>=2.14
```

### Pretrained Checkpoints

| Model | Description | Link |
|---|---|---|
| STAR-LDM | GPT-2 Large + diffusion planning, trained on FineWeb 100B | [Download](https://cornell.box.com/s/09kp1l61cmnejixpywqvg5vauoq8sih1) |
| Sentiment Classifier | Noise-conditioned MLP for sentiment-guided generation | [Download](https://cornell.box.com/s/gukku7f1k14vjteiqjrqz7y033ept58w) |

Download and extract each checkpoint to a local directory, e.g. `checkpoints/star-ldm/` and `checkpoints/sentiment-classifier/`.

## Generation

The generation script supports batch generation, interactive mode, and classifier-guided generation.

### Basic generation

```bash
python -m scripts.generate \
    --model_path checkpoints/star-ldm \
    --prompts "The movie was" "Once upon a time"
```

### Interactive mode

```bash
python -m scripts.generate \
    --model_path checkpoints/star-ldm \
    --interactive
```

### Classifier-guided generation

Steer generation toward a target attribute (e.g. positive or negative sentiment) using a pretrained classifier:

```bash
# Positive sentiment (cls_target=1.0)
python -m scripts.generate \
    --model_path checkpoints/star-ldm \
    --classifier_path checkpoints/sentiment-classifier \
    --cls_guidance 3.0 --cls_target 1.0 \
    --prompts "The movie was"

# Negative sentiment (cls_target=0.0)
python -m scripts.generate \
    --model_path checkpoints/star-ldm \
    --classifier_path checkpoints/sentiment-classifier \
    --cls_guidance 3.0 --cls_target 0.0 \
    --prompts "The movie was"
```

### Generation options

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Path to STAR-LDM checkpoint directory |
| `--prompts` | — | One or more prompts |
| `--interactive` | false | Enter interactive REPL mode |
| `--classifier_path` | — | Path to classifier checkpoint for guided generation |
| `--cls_guidance` | 0.0 | Classifier guidance scale (0 = disabled) |
| `--cls_target` | — | Target class for guidance (0.0 or 1.0) |
| `--sampling_timesteps` | 50 | Number of diffusion sampling steps |
| `--sampler` | ddpm | Diffusion sampler (`ddpm` or `ddim`) |
| `--cls_free_guidance` | 1.0 | Classifier-free guidance scale |
| `--max_new_tokens` | 64 | Maximum tokens to generate |
| `--top_p` | 0.9 | Nucleus sampling threshold |
| `--repetition_penalty` | 1.2 | Repetition penalty |

## Training

Train STAR-LDM on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) using streaming from the HuggingFace Hub (no data preprocessing required).

### Single GPU

```bash
python -m scripts.train --config configs/train_fineweb.yaml
```

### Multi-GPU (with Accelerate)

```bash
PYTHONPATH=. accelerate launch scripts/train.py --config configs/train_fineweb.yaml
```

### Override config values from the command line

```bash
PYTHONPATH=. accelerate launch scripts/train.py --config configs/train_fineweb.yaml \
    train.learning_rate=1e-4 train.train_batch_size=8
```

See [configs/train_fineweb.yaml](configs/train_fineweb.yaml) for the full set of training options. Key defaults match the pretrained checkpoint: GPT-2 Large backbone, WSD learning rate schedule, cosine noise schedule, sigmoid loss weighting, and 250K training steps on FineWeb `sample-10BT`.

## Architecture

STAR-LDM has three main components:

1. **Autoregressive Decoder** — GPT-2 Large (770M params). Generates tokens conditioned on soft prompt embeddings.
2. **Soft Prompt Generator** — Takes denoised sentence embeddings + timestep, produces soft prompt tokens injected into the LM context. Uses a time-conditioned Transformer with FiLM layers.
3. **Score Network Head** — Denoises sentence embeddings via iterative diffusion (v-prediction parameterization). Uses a separate Transformer with time conditioning.

The sentence embedding space is **Sentence-T5 XL** (768-dim). Total trainable parameters: ~956M.

**Generation procedure:**
1. Tokenize the text prefix into GPT-2 token embeddings
2. Diffusion sampling: starting from Gaussian noise, iteratively denoise a sentence embedding in Sentence-T5 space — at each step the noised embedding is transformed into soft prompt tokens, passed through GPT-2 with the prefix, and the score network predicts the denoising update (DDPM/DDIM)
3. The final denoised embedding is transformed into soft prompt tokens by the soft prompt generator
4. GPT-2 generates tokens autoregressively conditioned on the prefix embeddings and soft prompts

## License

This project is released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{
lovelace2025stopthinkautoregress,
title={Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning},
author={Justin Lovelace and Christian K Belardi and Sofian Zalouk and Adhitya Polavaram and Srivatsa R Kundurthy and Kilian Q Weinberger},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=c05qIG1Z2B}
}
```
