# Deep Learning Project 2 - AG News Classification with LoRA

## Project Overview
This project implements a modified BERT architecture using Low-Rank Adaptation (LoRA) for text classification on the AG News dataset. The goal is to achieve high test accuracy while maintaining the model under 1 million trainable parameters.

## Requirements
- Python 3.10+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- Scikit-learn

## Installation
```bash
pip install transformers datasets evaluate accelerate peft trl bitsandbytes
pip install nvidia-ml-py3
pip install scikit-learn
```

## Project Structure
```
.
├── final/
│   ├── CodeZero.ipynb        # Main implementation notebook
│   ├── test_unlabelled.pkl   # Test dataset
│   ├── submission 1.csv      # Submission file
│   ├── CodeZero_Report.pdf   # Project report
│   └── Code_Output.pdf       # Code execution output
```

## Implementation Details
- Base Model: RoBERTa
- Adaptation Method: LoRA (Low-Rank Adaptation)
- Dataset: AG News Classification
- Parameter Constraint: < 1 million trainable parameters
- LoRA Configuration Overview:
  - Rank (r): 7
  - Alpha: 16
  - Dropout: 0.05
  - Target Modules: query, key, value layers
  - Task Type: Sequence Classification

## Key Features
- LoRA implementation with configurable rank and alpha parameters
- Parameter-efficient fine-tuning (only 0.78% of parameters are trainable)
- Total trainable parameters: 980,740
- Hyperparameter optimization
- Test accuracy tracking
- Parameter count verification

## Results
- Final Test Accuracy: 88.91%
- Total Trainable Parameters: 980,740
- Model Architecture Details: RoBERTa base model with LoRA adaptation (r=7, alpha=16) applied to attention query, key, and value matrices in transformer layers. The model achieves parameter efficiency by making only 0.78% of the total parameters trainable.

### Classification Performance
Our model achieved strong performance across all four categories of the AG News dataset, as shown in Figure 5. The Sports category demonstrated the highest performance with an F1-score of 0.956, followed by Business (0.873), World (0.870), and Sci/Tech (0.859). The model maintained balanced precision and recall metrics across categories, with an overall accuracy of 88.91%. Notably, the class distribution analysis shows relatively balanced prediction counts across categories (140-186 samples per class), indicating robust performance without significant class bias. The macro-averaged F1-score of 0.890 suggests consistent performance across all categories, while the weighted average of 0.889 confirms the model's reliability considering class distributions.

## Report Contents
1. Project overview and findings summary
2. Methodology section
   - Model design process
   - Training approach
   - Hyperparameter choices and rationale
   - Lessons learned
3. Results section
   - Final test accuracy
   - Model architecture
   - Parameter count
4. Citations

## Repository Guidelines
- Code is well-documented
- Results are reproducible
- Notebooks include clear visualizations
- Parameter count and test accuracy are clearly printed