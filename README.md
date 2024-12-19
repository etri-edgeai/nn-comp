# nn-comp

**nn-comp** is a comprehensive framework for neural network compression and optimization, offering tools for efficient training, model distillation, low-rank approximation, pruning, and augmentation pipelines for EdgeAI.

## Features

- **Model Compression**: Includes low-rank approximations and pruning techniques.
- **Augmentation Tools**: Supports AutoAugment and RandAugment policies for image preprocessing.
- **Backend Support**: Seamlessly integrates with TensorFlow and PyTorch backends.
- **Custom Solvers**: Implements solvers like simulated annealing for optimization tasks.
- **DALI Integration**: Optimized data loading and augmentation using NVIDIA DALI.

## Directory Structure

```
./
├── examples
│   └── image_classification
│       ├── dataloader
│       ├── models
│       ├── utils
│       ├── _run.py
│       └── train.py
├── nncompress
│   ├── algorithms
│   ├── backend
│   ├── compression
│   ├── distillation
│   ├── handler
│   ├── search
│   ├── tools
│   ├── utils
├── tests
└── setup.py
```

### Key Components

- **`examples/image_classification`**: Contains scripts and modules for training image classification models.
  - `dataloader`: Data loading and preprocessing utilities.
  - `models`: Model architectures including ResNet, EfficientNet, ViT, and more.
  - `utils`: Helper functions for callbacks, learning rates, and optimizers.

- **`nncompress`**: Core compression framework.
  - `algorithms`: Optimization algorithms such as solvers.
  - `backend`: TensorFlow and PyTorch-specific backend implementations.
  - `compression`: Low-rank approximations and pruning utilities.
  - `distillation`: Model distillation support.
  - `search`: Tools for search and projection.

- **`tests`**: Unit tests for various components.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nn-comp.git
   cd nn-comp
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package:
   ```bash
   python setup.py install
   ```

## Usage

### Training an Image Classification Model

```bash
python examples/image_classification/train.py \
    --model resnet50 \
    --dataset imagenet \
    --batch-size 128 \
    --epochs 50
```

### Applying Model Compression

```python
from nncompress.compression import Pruner

# Apply pruning to a model
pruner = Pruner(model)
compressed_model = pruner.apply_pruning()
```

### Using Augmentation Policies

```python
from nncompress.backend.tensorflow_.data.augmenting_generator import AutoAugment

augmenter = AutoAugment(augmentation_name="v0")
augmented_image = augmenter.distort(input_image)
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes and commit: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

## Acknowledgments

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-00907, Development of Adaptive and Lightweight Edge-Collaborative Analysis Technology for Enabling Proactively Immediate Response and Rapid Learning).