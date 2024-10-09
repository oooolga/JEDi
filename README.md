<p align="center">
<a href="https://oooolga.github.io/jedi.github.io/">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/oooolga/JEDi/blob/main/assets/logo-dark_mode.png?raw=true" height="80">
        <source media="(prefers-color-scheme: light)" srcset="https://oooolga.github.io/JEDi.github.io/static/images/logo-sparkles.png" height="80">
        <img alt="logo" src="https://oooolga.github.io/JEDi.github.io/static/images/logo-sparkles.png" height="80">
    </picture>
</a>
</p>

# Beyond FVD: Enhanced Evaluation Metrics for Video Generation Quality

<p align="left">
<a href="https://arxiv.org/abs/2410.05203" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2410.05203-b31b1b.svg?style=flat" /></a>
<a href="https://oooolga.github.io/JEDi.github.io/" alt="webpage">
    <img src="https://img.shields.io/badge/Webpage-JEDi-darkviolet" /></a>
<img src="https://img.shields.io/github/license/oooolga/JEDi" />
<a href="https://paperswithcode.com/paper/beyond-fvd-enhanced-evaluation-metrics-for">
    <img alt="Static Badge" src="https://img.shields.io/badge/paper_with_code-link-turquoise?logo=paperswithcode" /></a>
<a href="https://pypi.org/project/VideoJEDi/" alt="webpage">
    <img src="https://img.shields.io/pypi/v/VideoJEDi" /></a>
<img src="https://views.whatilearened.today/views/github/oooolga/JEDi.svg" />
<p align="center">
<picture>
  <img src="https://oooolga.github.io/JEDi.github.io/static/images/teaser_plot.png">
</picture>
</p>

## Installation
Install our package from PyPI
```bash
pip install --upgrade videojedi
```

## Tutorial
Our metric has a simple setup process. To compute V-JEPA features and the JEDi metric, follow these simple steps:
- Pixel values are in the range [0, 1]
- Input dimension order is T, 3, H, W (frames, channels, height, width)

```python
from videojedi import JEDiMetric
jedi = JEDiMetric(feature_path=..., model_dir=...)
jedi.load_features(loaderA, loaderB, num_samples=...)
print(f"JEDi Metric: {jedi.compute_metric()}")
```

If V-JEPA features are already precomputed, simply load them and compute the JEDi metric.

```python
from videojedi import JEDiMetric
import numpy as np
jedi = JEDiMetric()
jedi.train_features = np.random.rand(5000, 1280)
jedi.test_features = np.random.rand(5000, 1280)
print(f"JEDi Metric: {jedi.compute_metric()}") 
```

Follow our interactive tutorial notebook for a detailed walkthrough: [Tutorial](https://github.com/oooolga/JEDi/blob/main/tutorials/JEDi_tutorial.ipynb).

## Citation

```bibtex
@misc{luo2024jedi,
        title={Beyond FVD: Enhanced Evaluation Metrics for Video Generation Quality}, 
        author={Ge Ya Luo and Gian Favero and Zhi Hao Luo and Alexia Jolicoeur-Martineau and Christopher Pal},
        year={2024},
        eprint={2410.05203},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2410.05203}, 
  }
```