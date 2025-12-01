# 🚀 Mini BERT (Encoder-Only) Implementation

A complete implementation of a simplified BERT model with encoder-only architecture for **Masked Language Modeling (MLM)** training. Built from scratch using PyTorch with proper backpropagation and checkpoint saving.

## 📋 Features

- ✅ **Complete Mini BERT Architecture**: Multi-head attention, positional encoding, layer normalization
- ✅ **Masked Language Modeling**: 15% token masking with BERT's masking strategy
- ✅ **Training with Backpropagation**: Full training loop with gradient clipping and optimization
- ✅ **Model Checkpointing**: Save and load trained models as `.pth` files
- ✅ **Interactive Evaluation**: Test your trained model on custom text
- ✅ **Attention Visualization**: Visualize what the model learns
- ✅ **Comprehensive Utilities**: Model analysis, embeddings export, benchmarking

## 🏗️ Architecture

```
Mini BERT Architecture:
├── Token Embeddings (vocab_size → d_model)
├── Positional Encoding (Sinusoidal)
├── Encoder Blocks (4 layers)
│   ├── Multi-Head Self-Attention (8 heads)
│   ├── Add & LayerNorm
│   ├── Feed-Forward Network
│   └── Add & LayerNorm
└── MLM Prediction Head (d_model → vocab_size)
```

**Default Configuration:**
- Model Dimension: 128
- Attention Heads: 8
- Encoder Layers: 4
- Feed-forward Dimension: 512
- Max Sequence Length: 128

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset
Place your `CRIME.TXT` file in the project directory, or the training script will create a sample dataset.

### 3. Train the Model
```bash
python train.py
```

This will:
- Load your crime dataset
- Build vocabulary from the text
- Train Mini BERT for 50 epochs
- Save checkpoints every 10 epochs
- Save the best model as `minibert_best_model.pth`
- Generate training plots

### 4. Evaluate the Model
```bash
# Evaluate on sample texts
python evaluate.py

# Interactive mode - test your own sentences
python evaluate.py --interactive

# Use specific checkpoint
python evaluate.py --checkpoint minibert_best_model.pth
```

## 📁 Project Structure

```
mini_bert/
├── tokenizer.py          # Simple BERT tokenizer with special tokens
├── dataset.py            # MLM dataset with masking strategy
├── model.py              # Mini BERT model architecture
├── train.py              # Training script with checkpointing
├── evaluate.py           # Evaluation and inference
├── utils.py              # Visualization and analysis tools
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🔧 Usage Examples

### Training Your Model
```python
from train import main
main()  # Trains model and saves checkpoints
```

### Loading and Using Trained Model
```python
from evaluate import MiniBERTEvaluator

# Load trained model
evaluator = MiniBERTEvaluator('minibert_best_model.pth')

# Predict masked tokens
text = "The detective investigated the [MASK] scene."
results = evaluator.predict_masked_sentence(text)
print(results)
```

### Model Analysis
```python
from utils import analyze_model_parameters, visualize_attention
from model import create_model

# Analyze model structure
model = create_model(vocab_size=1000)
analyze_model_parameters(model)

# Visualize attention (after training)
# visualize_attention(attention_weights, tokens)
```

## 📊 Model Checkpoints

The training script saves several types of checkpoints:

1. **`minibert_checkpoint_epoch_N.pth`** - Full checkpoint with optimizer state
2. **`minibert_best_model.pth`** - Best model state dict only
3. **`minibert_final_checkpoint.pth`** - Final complete checkpoint
4. **`minibert_model_state.pth`** - Final model state dict only

### Checkpoint Contents:
- Model state dictionary
- Optimizer state
- Training epoch and loss
- Tokenizer vocabulary
- Model configuration

## 🎯 Masked Language Modeling

The model uses BERT's masking strategy:
- **80%** of masked tokens → `[MASK]`
- **10%** of masked tokens → Random token
- **10%** of masked tokens → Original token (unchanged)

Example:
```
Input:  "The detective investigated the [MASK] scene."
Output: "crime" (0.65), "murder" (0.23), "accident" (0.08)
```

## 🔍 Evaluation Features

### Interactive Mode
```bash
python evaluate.py --interactive
```
Test your model interactively:
```
Enter text (with [MASK]): The suspect was [MASK] yesterday.
Predictions:
  1. arrested (0.823)
  2. caught (0.134)
  3. seen (0.043)
```

### Batch Evaluation
Evaluate on multiple samples automatically with crime-related test cases.

## 📈 Training Monitoring

The training script provides:
- Real-time loss tracking with progress bars
- Sample predictions every 5 epochs
- Training history plots
- Model parameter analysis
- Automatic best model saving

## 🛠️ Customization

### Modify Model Configuration
```python
config = {
    'd_model': 256,        # Increase model dimension
    'num_heads': 16,       # More attention heads
    'num_layers': 6,       # Deeper model
    'd_ff': 1024,          # Larger feed-forward
    'dropout': 0.2         # Higher dropout
}
```

### Custom Dataset
Replace `CRIME.TXT` with your own text file, or modify `dataset.py` to load different formats.

### Training Parameters
Modify `train.py` configuration:
```python
config = {
    'batch_size': 16,      # Larger batches
    'learning_rate': 5e-5, # Different learning rate
    'num_epochs': 100,     # More training epochs
    'save_every': 5        # Save more frequently
}
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib
- tqdm
- seaborn (for visualizations)

## 🎯 Example Results

After training on crime-related text:

```
Input: "The detective investigated the [MASK] scene."
Predictions:
  1. crime (0.745)
  2. murder (0.183)
  3. accident (0.072)

Input: "The suspect was [MASK] after the investigation."
Predictions:
  1. arrested (0.823)
  2. caught (0.134)
  3. detained (0.043)
```

## 🔬 Advanced Features

### Attention Visualization
```python
from utils import visualize_attention
# Visualize attention patterns (requires trained model)
```

### Token Similarity Analysis
```python
from utils import compute_token_similarities
# Find similar tokens in embedding space
```

### Model Benchmarking
```python
from utils import benchmark_model_speed
# Measure inference speed
```
