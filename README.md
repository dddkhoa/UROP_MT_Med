# English-Vietnamese Medical Translation Model

This project implements a fine-tuned language model for translating medical text from English to Vietnamese using the Qwen-2.5-7B model with LoRA (Low-Rank Adaptation) fine-tuning.

## Overview

The project uses the Unsloth-optimized Qwen-2.5-7B model and implements supervised fine-tuning (SFT) for English to Vietnamese translation. The model is optimized for efficiency using LoRA and includes comprehensive evaluation using BLEU scores.

## Requirements

### System Requirements
- Python 3.12
- CUDA 12.1
- PyTorch 2.4.0

### Python Dependencies
- transformers
- unsloth
- datasets
- evaluate
- hydra-core
- omegaconf

## Setup Environment

1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies**
   ```bash
   # Install project dependencies
   pip install -r requirements.txt

   # Install the project in editable mode
   pip install -e .
   ```

   Note: Make sure you have CUDA 12.1 installed on your system for GPU support.

## Project Structure

```
.
├── data/
│   ├── train.en.txt      # English training data
│   ├── train.vi.txt      # Vietnamese training data
│   ├── val.en.txt        # English validation data
│   ├── val.vi.txt        # Vietnamese validation data
│   ├── test.en.txt       # English test data
│   └── test.vi.txt       # Vietnamese test data
├── src/
│   └── mt_med/
│       ├── configs/
│       │   └── qwen_2.5.yaml  # Model configuration
│       └── train.py           # Training script
```

## Configuration

The model configuration is defined in `configs/qwen_2.5.yaml`:

```yaml
model:
  name: unsloth/Qwen2.5-7B
  rank: 16  # LoRA rank

load_in_4bit: True
max_seq_length: 512
```

## Training Pipeline

The training pipeline consists of several key steps:

1. **Model Initialization**
   - Loads the Qwen-2.5-7B base model
   - Configures model parameters including sequence length and quantization

2. **LoRA Setup**
   - Implements parameter-efficient fine-tuning using LoRA
   - Targets key attention modules for adaptation
   - Uses rank 16 for efficiency

3. **Data Processing**
   - Uses an Alpaca-style prompt template for translation
   - Processes parallel English-Vietnamese text data
   - Implements efficient data loading and formatting

4. **Training Configuration**
   - Batch size: 16 (per device)
   - Gradient accumulation steps: 4
   - Learning rate: 3e-4
   - Training epochs: 1
   - Maximum steps: 60
   - Uses 8-bit AdamW optimizer
   - Implements mixed precision training (FP16/BF16)

5. **Evaluation**
   - Performs validation during training
   - Calculates SacreBLEU scores on test set (using HuggingFace's evaluate library)
   - Generates translations with max 64 new tokens

## Usage

1. **Setup Environment**
   ```bash
   # Install required packages
   pip install torch transformers unsloth datasets nltk hydra-core omegaconf
   ```

2. **Prepare Data**
   - Place your parallel English-Vietnamese text files in the `data/` directory
   - Files should be named as specified in the project structure

3. **Training**
   ```bash
   # Run training with default configuration (qwen_2.5)
   python -m mt_med.train

   # Run training with a specific configuration
   python -m mt_med.train --config-name [CONFIG_NAME]
   ```
   
   The training script uses Hydra for configuration management. You can:
   - Use the default configuration (`qwen_2.5.yaml`)
   - Specify a different configuration file using `--config-name`
   - Override specific configuration values using command line arguments

   Example:
   ```bash
   # Use a different configuration file
   python -m mt_med.train --config-name qwen_2.5_instruct

   # Override specific configuration values
   python -m mt_med.train model.rank=32 max_seq_length=1024
   ```

4. **Model Outputs**
   - Fine-tuned model will be saved in the specified output directory
   - Training logs include:
     - Training statistics
     - Validation metrics
     - Final SacreBLEU scores

## Model Features

- **Efficient Training**
  - Uses 4-bit quantization
  - Implements LoRA for parameter-efficient fine-tuning
  - Supports gradient checkpointing
  - Optimized for memory efficiency

- **Evaluation Metrics**
  - Real-time validation during training
  - SacreBLEU score calculation on test set (industry standard metric)
  - Detailed logging of training statistics

## Experimental Results

### 1. Direct In-context Prompting (DIPMT) Results
BLEU scores tested on 500 sentences from MedEV, with few-shots selected randomly:

| Model        | 0-shot (e-v) | 0-shot (v-e) | 1-shot (e-v) | 1-shot (v-e) | 3-shot (e-v) | 3-shot (v-e) |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Qwen2.5-0.5B | 9.56        | 12.26       | 12.00       | 11.42       | 11.56       | 10.54       |
| Qwen2.5-3B   | 27.40       | 22.14       | 26.50       | 20.69       | 26.00       | 19.94       |
| Qwen2.5-7B   | 32.69       | 23.28       | 32.86       | 24.90       | 32.27       | 24.30       |

### 2. LoRA Fine-tuning Results
BLEU scores trained on trainset of MedEV, tested on 100 test sentences:

| Model                           | BLEU Score |
|--------------------------------|------------|
| Qwen2.5-7B                     | 14.93      |
| Qwen2.5-7B-Finetuned          | 28.41      |
| Qwen2.5-7B-Instruct           | 12.47      |
| Qwen2.5-7B-Instruct-Finetuned | 25.50      |
| Llama3.2-8B-Instruct          | 13.91      |
| Llama3.2-8B-Instruct-Finetuned| 27.03      |

**Note**: Improvements from fine-tuning mostly come from model following output format in fine-tuning prompt, not actual translation capability.

Number of trainable parameters = 40,370,176 = 0.5% of original model size

### 3. Multiturn Prompting Results
BLEU scores tested on 500 sentences from MedEV, using Llama3.1-8B-Instruct:

| Method              | Overall BLEU |
|--------------------|--------------|
| Direct Prompting   | 38.13        |
| Multiturn Prompting| 36.86        |

## Notes

- The model requires a Hugging Face token for downloading
- Training is optimized for CUDA-enabled devices
- The implementation supports both FP16 and BF16 precision based on hardware capability
- Evaluation uses SacreBLEU from HuggingFace's evaluate library for industry-standard BLEU score calculation

## License

[Specify your license here]

## Citation

[Add any relevant citations or references]
