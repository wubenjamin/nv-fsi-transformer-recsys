# NVIDIA FSI Transformer RecSys Demo

This repository contains demonstration notebooks showcasing the use of [NVIDIA Merlin Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec/) for building sequential and session-based recommendation systems using transformer architectures, specifically adapted for **Financial Services Industry (FSI)** use cases.

## Overview

Transformers4Rec is a flexible and efficient library for sequential and session-based recommendation that works with PyTorch. This demo repository demonstrates how to leverage GPU-accelerated preprocessing with NVTabular and transformer models for **financial product recommendation** and **loan customer journey modeling** in the financial services industry.

## Notebooks

### 1. FSI Loan Data ETL with NVTabular (`01-FSI-Loan-Data-ETL-with-NVTabular.ipynb`)
This notebook demonstrates:
- **FSI-specific data preprocessing**: Loading and processing synthetic loan interaction data
- **Financial feature engineering**: Creating product interaction sequences, customer financial profiles, and eligibility features
- **GPU-accelerated transformations**: Optimized preprocessing for financial datasets using NVTabular
- **Schema creation for FSI data**: Feature preparation specifically designed for loan customer journeys
- **Integration with financial data**: Support for FICO scores, income data, loan details, and marketing touchpoints

### 2. FSI Financial Product Recommendation with XLNet (`02-FSI-Financial-Product-Recommendation-XLNet.ipynb`)
This notebook demonstrates:
- **Transformer Integration**: Seamless integration with Hugging Face Transformers for RecSys tasks
- **Financial product recommendation**: Building recommendation models for financial services using transformers
- **Loan customer journey modeling**: Sequential modeling of customer interactions with financial products and services
- **Next-product prediction**: Predicting the next financial product or service a customer is likely to engage with
- **FSI-specific evaluation metrics**: Tailored evaluation approach for financial recommendation systems
- **Performance optimization**: Techniques optimized for financial product vocabularies and customer sequences

## Key Features

- **FSI Domain Adaptation**: Specifically adapted for Financial Services Industry use cases
- **GPU-Accelerated Pipeline**: Leverages NVIDIA Merlin components for end-to-end GPU acceleration
- **Financial Product Focus**: Specialized for loan products, financial services, and customer journey modeling
- **Rich Financial Features**: Support for FICO scores, income data, loan characteristics, eligibility criteria, and marketing interactions
- **Customer Journey Analytics**: Session-based modeling of how customers interact with financial products over time

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
4. Start with `01-FSI-Loan-Data-ETL-with-NVTabular.ipynb` for FSI data preprocessing and feature engineering
5. Continue with `02-FSI-Financial-Product-Recommendation-XLNet.ipynb` for financial product recommendation model training

## FSI Use Case Details

This demo specifically models:
- **Loan Customer Journeys**: Sequential interactions of customers with loan products and services
- **Financial Product Recommendations**: Predicting which financial products customers are likely to engage with next
- **Customer Segmentation**: Using financial profiles (FICO, income, loan details) for personalized recommendations
- **Marketing Optimization**: Incorporating marketing touchpoint data (email, direct mail) into recommendation logic

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