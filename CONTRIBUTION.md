# Contributing to Waste Classification System

## Code Contributors

**Ayenor Oghenefejiro Christopher** & **Oluwateniola Ajetomobi**

**Collaborative Contributions:**

- Dataset collection, preparation, and splitting
- Project design and methodology development
- Comprehensive 30-page research report
- Model architecture selection and justification
- Results analysis and interpretation

**Code Development:**

- Model training pipeline (Ayenor - lead)
- Documentation and usage guidelines (Oluwateniola - lead)

## Development Notes

### Dataset Preparation

**Dataset Statistics:**

- Total images: 34,459
- Recyclables: 17,677 (51.3%)
- Non-Recyclables: 16,782 (48.7%)

**Split Strategy:**

- Training: 70% (~24,121 images)
- Validation: 15% (~5,169 images)
- Test: 15% (~5,169 images)
- Stratified splitting maintains class balance across all sets

**Data Collection:**

- Custom images captured in various lighting and angles
- Online open-source datasets for diversity
- Manual labeling and quality verification

### Two-Phase Training Strategy

Our transfer learning approach uses two distinct phases:

**Phase 1: Feature Extraction**

- Base MobileNetV2 layers are frozen
- Only the classification head is trained
- Higher learning rate (0.001) for faster convergence
- Goal: Teach the head to use ImageNet features for waste classification

**Phase 2: Fine-Tuning**

- Last 30 layers of MobileNetV2 are unfrozen
- Lower learning rate (0.0001) to preserve pre-trained weights
- Allows high-level features to adapt to waste-specific patterns
- Goal: Optimize entire model for recyclable/non-recyclable distinction

### Model Architecture Decisions

**Why MobileNetV2?**

- Lightweight (only 2.4M parameters total)
- Efficient for edge deployment (smart bins, mobile devices)
- Fast inference (<50ms per image)
- Pre-trained on ImageNet provides strong feature extraction

**Why Binary Classification?**

- Foundation for more complex multi-class systems
- Simplifies deployment in smart bins
- Achieves high accuracy (97%+) with smaller dataset
- Easier to explain and debug

### Model Performance

**Test Set Results:**

- Accuracy: 97.31%
- Precision: ~97.7%
- Recall: ~97.0%
- AUC-ROC: 0.9960

**Error Analysis:**

- False Positives: 58 (2.3%)
- False Negatives: 81 (3.0%)
- Both error rates are minimal for real-world deployment

## Usage Guidelines

### For Training

```bash
# Ensure dataset is split first
python scripts/split_dataset.py

# Then train the model
cd src
python train_and_evaluate.py
```

### For Predictions

```bash
# Place test images in input_images/ folder
python predict.py
```

### Confidence Score Interpretation

- \*>90%: Very high confidence - model is very certain
- 75-90%: High confidence - reliable prediction
- 60-75%: Moderate confidence - generally accurate
- <60%: Low confidence - consider manual verification

## Project Structure

```
waste-classification/
├── data/                  # Generated dataset splits
├── DatasetFinal/          # Original dataset
├──input_images/          # Test images for prediction
├── models/                # Trained model files
├── results/               # Evaluation visualizations
├── scripts/               # Utility scripts
├── src/                   # Main training code
├── predict.py             # Prediction script
└── README.md              # Main documentation
```

## Real-World Deployment

This model is designed for integration with:

- Smart waste bins with embedded cameras
- Mobile applications for recycling guidance
- Automated sorting systems in recycling facilities
- Educational tools for waste management awareness
