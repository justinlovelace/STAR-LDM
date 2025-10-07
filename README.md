# STAR-LDM: Stop-Think-AutoRegress Language Diffusion Model

**[Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning](https://openreview.net/forum?id=c05qIG1Z2B)**

Justin Lovelace, Christian Belardi, Sofian Zalouk, Adhitya Polavaram, Srivatsa Kundurthy, Kilian Q Weinberger

_Conference on Language Modeling (COLM) 2024_

---

![STAR-LDM Overview](fig/starldm.png)

## Abstract

STAR-LDM integrates latent diffusion planning with autoregressive generation. Unlike conventional autoregressive language models limited to token-by-token decisions, STAR-LDM incorporates a 'thinking' phase that pauses generation to refine a semantic plan through diffusion before continuing. The model enables global planning in continuous space prior to committing to discrete tokens and demonstrates significant performance improvements in language understanding benchmarks.

## Code Release

Code will be released soon.

## Citation

```bibtex
@inproceedings{lovelace2024stopthink,
  title={Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning},
  author={Lovelace, Justin and Belardi, Christian K and Zalouk, Sofian and Polavaram, Adhitya and Kundurthy, Srivatsa R and Weinberger, Kilian Q},
  booktitle={Conference on Language Modeling (COLM)},
  year={2024},
  url={https://openreview.net/forum?id=c05qIG1Z2B}
}
```