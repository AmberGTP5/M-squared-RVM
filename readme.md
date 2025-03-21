# Multimodal Math Reasoning and Verification Model

This project develops a multimodal vision-language model focused on mathematical reasoning and step verification. The project follows a three-stage training process, from foundational pretraining to fine-tuning for mathematical reasoning, ultimately creating a model capable of verifying reasoning steps.

## Project Structure

### 1. Foundational Vision-Language Model (VLM) - `train.py`
The base model architecture integrates vision and language capabilities:
- **Vision Model**: SigLIP-base-patch16-224
- **Language Model**: Qwen2.5-0.5B-Instruct
- **Connection Mechanism**: Uses a linear layer to map visual features to the language space
- **Training Data**: LLaVA-CC3M-Pretrain-595K
- **Features**:
  - Implements multimodal fusion of images and text
  - Supports `<|image_pad|>` tokens for image embedding
  - Optimizes memory usage and training speed

### 2. Supervised Fine-Tuning (SFT) - `sft_train.py`
Fine-tuning for geometry problem-solving:
- **Training Data**: Geo170K geometry problem dataset
- **Fine-tuning Strategy**:
  - Freezes the vision model and linear projection layer
  - Fine-tunes only the language model
  - Enhances the model’s ability to answer geometry questions
  - Implements a specialized dialogue template processing

### 3. Proof Process Model (PPM) - `ppm_train.py`
Enhances the model’s ability to verify reasoning steps:
- **Core Innovation**: Introduces a verification head to assess reasoning step correctness
- **Training Data**: DualMath-1.1M dataset
- **Model Extension**:
  - Adds a `BinaryPredictionHead` for binary classification
  - Identifies and extracts reasoning steps
  - Jointly trains generation and verification capabilities
- **Features**:
  - Uses gradient checkpointing to save memory
  - Designed data loading and processing pipeline for DualMath format
  - Custom loss function combining generation and verification tasks

## Dataset Acquisition and Preparation

### Pretraining Datasets
- [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)
- [Chinese Translation Version](https://huggingface.co/datasets/chinese-llava/Chinese-LLaVA-Vision-Instructions)

### Geometry Problem Dataset
- [Geo170K](https://huggingface.co/datasets/geometer/Geo170K)

### Mathematical Verification Datasets
- [MathV360K](https://huggingface.co/datasets/BAAI/MathV360K)
- [DualMath-1.1M](https://huggingface.co/datasets/TIGER-Lab/DualMath-1.1M)

## Model Download and Usage

### Base Models
- **Vision Model**: [SigLIP-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224)
- **Language Model**: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## Training and Inference
- **Training Process**
- **Model Inference**

## Key Technical Highlights
1. **Multi-Stage Training Strategy**: From general pretraining to task-specific fine-tuning
2. **Parameter Freezing and Fine-Tuning**: Different strategies applied in various training stages
3. **Memory Optimization**: Techniques like gradient checkpointing and mixed-precision training
4. **Modular Design**: Facilitates model extension and modification
5. **Custom Data Processing**: Supports various dataset formats
6. **Verification Capability**: Not only generates mathematical solutions but also verifies reasoning steps

## Training Configuration
- **Hardware**: CUDA GPU acceleration
- **Batch Size**: Adjusted per stage (4-16)
- **Learning Rate**: Decays from 5e-4 to 3e-5
- **Optimizer**: Uses Transformers' default optimizer
- **Training Epochs**: Varies from 2-5 epochs per stage
- **Checkpointing**: Regularly saves checkpoints, keeping the best model

## Application Scenarios
- Solving image-based mathematical and geometry problems
- Verifying mathematical reasoning steps
- Educational assistance: Provides reasoning steps and highlights errors
- Intelligent tutoring system: Helps students understand mathematical reasoning

## System Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-supported GPU (16GB+ VRAM recommended)
- At least 32GB system RAM

## References
1. Liu, H., et al. (2023). "Visual Instruction Tuning." arXiv preprint arXiv:2304.08485.
2. Li, C., et al. (2023). "Qwen Technical Report." arXiv preprint arXiv:2309.16609.
3. Chen, J., et al. (2023). "SigLIP: Signal Language Image Pre-training." arXiv preprint arXiv:2303.15343.
4. Zheng, C., et al. (2022). "Geometry Problem Solving by Integrating Language and Vision." arXiv preprint arXiv:2211.16374.
5. Liu, X., et al. (2023). "DualMath: Dual Verification and Reasoning for Language Model-based Mathematical Problem Solving." arXiv preprint arXiv:2310.11350.
6. Zhang, H., et al. (2023). "MathVista: Evaluating Mathematical Reasoning of Visual-Language Models." arXiv preprint arXiv:2310.02255.

## Acknowledgments
- Thanks to Hugging Face for providing the Transformers library and model hosting.
- Appreciation to Google Research for developing the SigLIP model.
- Gratitude to the Qwen team for the Qwen model series.
- Thanks to the contributors of the datasets used for training.
- Special thanks to computing resource providers for their support.

