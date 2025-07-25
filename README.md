# Introducing SDICE: An Index for Assessing Diversity of Synthetic Medical Datasets

[![Paper](https://img.shields.io/badge/Paper-Available-green)]([https://github.com/your-repo/sdice](https://bmva-archive.org.uk/bmvc/2024/workshops/PFATCV/12_Introducing_SDICE_An_Index_.pdf))
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-yellow)](https://github.com/your-repo/sdice)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Authors

**Mohammed Talha Alam**, **Raza Imam**, **Mohammad Areeb Qazi**, **Asim Ukaye**, **Karthik Nandakumar**

Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, United Arab Emirates

## Abstract

Advancements in generative modeling are pushing the state-of-the-art in synthetic medical image generation. These synthetic images can serve as an effective data augmentation method to aid the development of more accurate machine learning models for medical image analysis. While the fidelity of these synthetic images has progressively increased, the diversity of these images is an understudied phenomenon. In this work, we propose the **SDICE index**, which is based on the characterization of similarity distributions induced by a contrastive encoder. Given a synthetic dataset and a reference dataset of real images, the SDICE index measures the distance between the similarity score distributions of original and synthetic images, where the similarity scores are estimated using a pre-trained contrastive encoder. This distance is then normalized using an exponential function to provide a consistent metric that can be easily compared across domains. Experiments conducted on the MIMIC-chest X-ray and ImageNet datasets demonstrate the effectiveness of SDICE index in assessing synthetic medical dataset diversity.

## Key Features

- 📊 **Novel Diversity Metric**: First comprehensive index specifically designed for evaluating diversity in synthetic medical datasets
- 🔬 **Contrastive Encoder Based**: Leverages pre-trained contrastive encoders to capture meaningful similarity distributions
- 📈 **Normalized Scoring**: Exponential normalization provides consistent metrics comparable across different domains
- 🏥 **Medical Focus**: Specifically addresses the unique challenges in synthetic medical image diversity assessment
- 🔍 **Intra/Inter-class Analysis**: Separately evaluates within-class and between-class diversity variations
- ⚖️ **Comparative Evaluation**: Outperforms traditional metrics like MS-SSIM and FID in medical image contexts

## 🚧 Repository Status

**This repository is currently under development. Code, models, and documentation will be uploaded soon.**

We are in the process of:
- [ ] Cleaning and organizing the codebase
- [ ] Preparing evaluation scripts for MIMIC-CXR and ImageNet
- [ ] Writing comprehensive documentation and tutorials
- [ ] Setting up pre-trained contrastive encoder models
- [ ] Creating example notebooks and usage guides

**Expected Release**: Coming Soon

## Methodology Overview

The SDICE index works by:

1. **Feature Extraction**: Uses pre-trained contrastive encoders to extract meaningful representations
2. **Similarity Distribution**: Computes similarity distributions for both intra-class and inter-class pairs
3. **Distance Measurement**: Calculates F-ratio between synthetic and real image similarity distributions
4. **Normalization**: Applies exponential normalization for domain-independent comparison

```
SDICE := (γ_intra, γ_inter)
γ = √(γ_intra² + γ_inter²)
```

Where higher values of γ indicate better diversity.

## Experimental Results

Our experiments demonstrate:

- **MIMIC-CXR Dataset**: Synthetic CXR images show significantly lower intra-class diversity (γ_intra = 0.11)
- **ImageNet Dataset**: Better overall diversity performance (γ_intra = 0.47)
- **Class-specific Analysis**: Domain-specific medical terms show poorer diversity than general concepts
- **Prompt Sensitivity**: Different prompt complexities significantly affect generated image diversity

## 📋 Requirements

The code will be released with detailed requirements. Expected dependencies include:
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- SciPy
- scikit-learn
- matplotlib
- seaborn
- Additional dependencies will be listed in `requirements.txt`

## 🚀 Quick Start

Once the code is released, you'll be able to:

```bash
# Clone the repository
git clone https://github.com/your-username/sdice.git
cd sdice

# Install dependencies
pip install -r requirements.txt

# Compute SDICE index for your datasets
python compute_sdice.py --synthetic_path /path/to/synthetic --real_path /path/to/real --output results.json

# Visualize diversity distributions
python visualize_results.py --results results.json
```

## 📊 Supported Datasets

The current implementation supports evaluation on:
- **MIMIC-CXR**: Chest X-ray images with 14 diagnostic labels
- **ImageNet**: Natural images with matched classes to MIMIC-CXR
- **FairFace**: Additional validation dataset (see supplementary material)
- **Custom Datasets**: Easy integration with your own synthetic/real image pairs

## 🔧 Customization Options

The SDICE index is highly configurable:
- **Feature Extractors**: Support for different pre-trained contrastive encoders
- **Similarity Metrics**: Cosine similarity, Euclidean distance, etc.
- **Distance Measures**: F-ratio, Earth Mover's Distance (EMD), KL-divergence
- **Normalization Functions**: Exponential, linear, or custom normalization schemes

## 📈 Comparison with Existing Metrics

| Metric | Domain-Specific | Normalized | Intra/Inter-class | Medical Focus |
|--------|----------------|------------|-------------------|---------------|
| MS-SSIM | ❌ | ❌ | ❌ | ❌ |
| FID | ❌ | ❌ | ❌ | ❌ |
| Vendi Score | ❌ | ✅ | ❌ | ❌ |
| **SDICE** | ✅ | ✅ | ✅ | ✅ |

## 📖 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{alam2024sdice,
    title={Introducing SDICE: An Index for Assessing Diversity of Synthetic Medical Datasets},
    author={Alam, Mohammed Talha and Imam, Raza and Qazi, Mohammad Areeb and Ukaye, Asim and Nandakumar, Karthik},
    journal={arXiv preprint},
    year={2024}
}
```

## 🤝 Contributing

We welcome contributions to improve the SDICE index! Areas where contributions are particularly valuable:
- Additional distance metrics implementation
- Support for new medical imaging modalities  
- Computational efficiency improvements
- Integration with popular generative model frameworks

## 📧 Contact

For questions about this work, please contact:

- **Mohammed Talha Alam**: mohammed.alam@mbzuai.ac.ae

## 📄 License

This project will be released under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

We thank the authors of RoentGen, UniDiffuser, and the MIMIC-CXR dataset for making their work publicly available, which enabled our comprehensive evaluation.

---

**Note**: This repository contains the implementation of the SDICE index for evaluating diversity in synthetic medical datasets. Stay tuned for the code release!
