# NVIDIA FSI Transformer RecSys Demo

This repository contains demonstration notebooks showcasing the use of [NVIDIA Merlin Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec/) for building sequential and session-based recommendation systems using transformer architectures.

## Overview

Transformers4Rec is a flexible and efficient library for sequential and session-based recommendation that works with PyTorch. This demo repository demonstrates how to leverage GPU-accelerated preprocessing with NVTabular and transformer models for recommendation tasks in the financial services industry (FSI).

## Notebooks

### 1. ETL with NVTabular (`01-ETL-with-NVTabular.ipynb`)
This notebook demonstrates:
- Data preprocessing and feature engineering using NVTabular
- GPU-accelerated data transformations for recommendation datasets
- Schema creation and feature preparation for transformer models
- Integration with the Merlin ecosystem for efficient data pipelines

### 2. Session-based XLNet with PyTorch (`02-session-based-XLNet-with-PyT.ipynb`)
This notebook demonstrates:
- Building session-based recommendation models using XLNet transformer architecture
- Training and evaluation of sequential recommendation models
- Next-item prediction tasks using Transformers4Rec
- Performance optimization techniques for transformer-based RecSys

## Key Features

- **GPU-Accelerated Pipeline**: Leverages NVIDIA Merlin components for end-to-end GPU acceleration
- **Transformer Integration**: Seamless integration with Hugging Face Transformers for RecSys tasks
- **Rich Feature Support**: Support for multiple input features beyond simple token sequences
- **Session-based Recommendation**: Specialized for sequential and session-based recommendation scenarios

## Requirements

To run these notebooks, you'll need:

- NVIDIA GPU with CUDA support
- Python 3.8+
- PyTorch with CUDA support
- NVIDIA Merlin components (Transformers4Rec, NVTabular)
- cuDF for GPU-accelerated DataFrame operations

## Installation

### Using Conda (Recommended)
```bash
mamba create -n transformers4rec-demo -c nvidia -c rapidsai -c pytorch -c conda-forge \
    transformers4rec=23.04 \
    nvtabular=23.04 \
    python=3.10 \
    cudf=23.02 \
    cudatoolkit=11.8 pytorch-cuda=11.8
```

### Using Pip
```bash
pip install transformers4rec[nvtabular]
pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
```

### Using Docker
```bash
docker run --gpus all -it -p 8888:8888 nvcr.io/nvidia/merlin/merlin-pytorch:23.04
```

## Getting Started

1. Clone this repository
2. Install the required dependencies
3. Launch Jupyter Lab/Notebook
4. Start with `01-ETL-with-NVTabular.ipynb` for data preprocessing
5. Continue with `02-session-based-XLNet-with-PyT.ipynb` for model training

## About NVIDIA Merlin

[NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) is an open-source library providing end-to-end GPU-accelerated recommender systems. Transformers4Rec is a key component that brings the power of transformer architectures to recommendation tasks.

## References

- [Transformers4Rec GitHub Repository](https://github.com/NVIDIA-Merlin/Transformers4Rec/)
- [Transformers4Rec Documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/)
- [NVIDIA Merlin Documentation](https://nvidia-merlin.github.io/Merlin/main/README.html)
- [End-to-end pipeline with NVIDIA Merlin](https://nvidia-merlin.github.io/Transformers4Rec/main/examples/end-to-end-session-based/)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This demo is built upon the excellent work of the NVIDIA Merlin team and the broader open-source community. Special thanks to the contributors of Transformers4Rec for making transformer-based recommendation systems accessible and efficient.

## Support

For questions about Transformers4Rec, please refer to:
- [Official Documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/)
- [GitHub Issues](https://github.com/NVIDIA-Merlin/Transformers4Rec/issues)
- [NVIDIA Merlin Community](https://developer.nvidia.com/merlin-devzone-survey) 