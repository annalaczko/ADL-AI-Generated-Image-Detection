# Archive

This folder contains earlier experimental notebooks kept for reference. They are **not part of the final pipeline** and do not need to be run.

| Notebook | Description |
|----------|-------------|
| `Neural Network 32.ipynb` | Early CNN experiment trained on 32×32 images |
| `Neural Network 64.ipynb` | CNN experiment on 64×64 images |
| `Neural Network 128.ipynb` | CNN experiment on 128×128 images (precursor to the final model) |
| `Neural Network 256.ipynb` | CNN experiment on 256×256 images |
| `Preprocess.ipynb` | Original preprocessing notebook before it was refactored into `Preprocess.py` |
| `Model_Visualization.ipynb` | Generates the `model_structure_high_quality.png/pdf` architecture diagrams using torchviz |

The progression from 32 → 128px showed that larger input resolution significantly improves classification accuracy, which motivated the final choice of 128×128 for the trained model.
