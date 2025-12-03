"""
Waste Classification Training Script
Uses MobileNetV2 with transfer learning for binary classification
"""
# Core libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

#==============================================================================#
# PATHS
#==============================================================================#

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'val')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create output directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

#==============================================================================#
# HYPERPARAMETERS
#==============================================================================#

print("="*70)
print("WASTE CLASSIFICATION SYSTEM")
print("Binary Classification: Recyclable vs Non-Recyclable")
print("="*70)

# Architecture
INPUT_SHAPE = (224, 224, 3)     # Standard MobileNetV2 input size
DENSE_UNITS = 128               # Size of classification layer
DROPOUT_RATE = 0.5              # Dropout to prevent overfitting

# Training Phase 1 (Feature Extraction with frozen base)
BATCH_SIZE = 32                 # Number of images processed together
EPOCHS_PHASE1 = 10              # Quick training of classification head
LEARNING_RATE_PHASE1 = 0.001    # Higher LR for training new layers

# Training Phase 2 (Fine-tuning with unfrozen layers)
EPOCHS_PHASE2 = 20              # More epochs for fine-tuning
LEARNING_RATE_PHASE2 = 0.0001   # Lower LR to preserve pretrained weights
UNFREEZE_LAYERS = 30            # Number of top layers to unfreeze

print("\nHyperparameters:")
print(f"  Input Shape: {INPUT_SHAPE}")
print(f"  Dense Units: {DENSE_UNITS}")
print(f"  Dropout Rate: {DROPOUT_RATE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Phase 1 Epochs: {EPOCHS_PHASE1}")
print(f"  Phase 2 Epochs: {EPOCHS_PHASE2}")
print(f"  Phase 1 LR: {LEARNING_RATE_PHASE1}")
print(f"  Phase 2 LR: {LEARNING_RATE_PHASE2}")
print(f"  Unfreeze Layers: {UNFREEZE_LAYERS}")

#==============================================================================#
# DATA PREPROCESSING AND AUGMENTATION
#==============================================================================#

print("\n" + "="*70)
print("DATA PREPROCESSING AND AUGMENTATION")
print("="*70)

# Check if data directories exist
if not os.path.exists(TRAIN_DIR):
    print(f"\nERROR: Training directory '{TRAIN_DIR}' not found!")
    print("Please run 'splitDataset.py' first to create the dataset splits.")
    sys.exit(1)

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Validation and test data (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nDataset Split:")
print(f"  Training: {train_generator.samples} images")
print(f"  Validation: {val_generator.samples} images")
print(f"  Test: {test_generator.samples} images")
print(f"  Class indices: {train_generator.class_indices}")
# ... (your existing code above) ...

print("\nData Augmentation (Training only):")
print("  - Rotation: ±20 degrees")
print("  - Width/Height shift: ±10%")
print("  - Shear: ±10%")
print("  - Zoom: 80-120%")
print("  - Horizontal flip: Yes")
print("  - Brightness: 80-120%")

#==============================================================================
# MODEL BUILDING
#==============================================================================

print("\n" + "="*70)
print("BUILDING MODEL")
print("="*70)

# Load pre-trained MobileNetV2
print("\nLoading MobileNetV2 (pre-trained on ImageNet)...")
base_model = MobileNetV2(
    input_shape=INPUT_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze all base model layers initially
base_model.trainable = False

# Add custom classification layers
print("Adding custom classification layers...")
x = base_model.output
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
x = Dense(DENSE_UNITS, activation='relu', name='dense')(x)
x = Dropout(DROPOUT_RATE, name='dropout')(x)
outputs = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=base_model.input, outputs=outputs, name='waste_classifier')

# Compile model for Phase 1
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

print(f"\nModel Summary:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

# Model Architecture Notes:
# MobileNetV2 base provides efficient feature extraction (~2.2M params)
# Custom head has only ~164K trainable params (keeps model lightweight)
# This architecture enables deployment on edge devices like Raspberry Pi
#==============================================================================
# PHASE 1: FEATURE EXTRACTION
#==============================================================================

print("\n" + "="*70)
print("PHASE 1: FEATURE EXTRACTION")
print("="*70)
print("Training classification head (base model frozen)")
print("This may take 15-30 minutes depending on your hardware...\n")

# Callbacks
callbacks_phase1 = [
    EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True, 
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(MODELS_DIR, 'model_phase1.h5'),  # ← Creates model_phase1.h5
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-7, 
        verbose=1
    )
]

# Train Phase 1
history_phase1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    callbacks=callbacks_phase1,
    verbose=1
)

print("\nPhase 1 completed!")

# Phase 1 Training Complete:
# - Classification head has learned to distinguish recyclable vs non-recyclable
# - Base model weights remain frozen (preserved ImageNet features)
# - Model is now ready for Phase 2 fine-tuning
#==============================================================================
# PHASE 2: FINE-TUNING
#==============================================================================

print("\n" + "="*70)
print("PHASE 2: FINE-TUNING")
print("="*70)

# Unfreeze last 30 layers
base_model.trainable = True
total_layers = len(base_model.layers)
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])

print(f"Base model has {total_layers} layers")
print(f"Unfrozen last {UNFREEZE_LAYERS} layers")
print(f"Trainable layers: {trainable_layers}")
print("This may take 45-90 minutes depending on your hardware...\n")

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# Callbacks
callbacks_phase2 = [
    EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True, 
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_model.h5'),  # ← Creates best_model.h5
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-7, 
        verbose=1
    )
]

# Train Phase 2
history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    verbose=1
)

# Save final model
model.save(os.path.join(MODELS_DIR, 'final_model.h5'))  # ← Creates final_model.h5
print(f"\nModels saved to '{MODELS_DIR}/' directory")

#==============================================================================
# TRAINING VISUALIZATION
#==============================================================================

print("\n" + "="*70)
print("GENERATING TRAINING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history_phase2.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history_phase2.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history_phase2.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history_phase2.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history_phase2.history['precision'], label='Train', linewidth=2)
axes[1, 0].plot(history_phase2.history['val_precision'], label='Validation', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history_phase2.history['recall'], label='Train', linewidth=2)
axes[1, 1].plot(history_phase2.history['val_recall'], label='Validation', linewidth=2)
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=300)  # ← Creates training_curves.png
print(f"Training curves saved to '{RESULTS_DIR}/training_curves.png'")
plt.close()

#==============================================================================
# MODEL EVALUATION ON TEST SET
#==============================================================================

print("\n" + "="*70)
print("MODEL EVALUATION ON TEST SET")
print("="*70)

# Get predictions
print("\nGenerating predictions...")
y_pred_proba = model.predict(test_generator, verbose=1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Classification Report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
class_names = ['Non-Recyclable', 'Recyclable']
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Overall Metrics
accuracy = np.mean(y_true == y_pred)
auc_score = roc_auc_score(y_true, y_pred_proba)

print("\n" + "="*70)
print("OVERALL METRICS")
print("="*70)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"AUC-ROC: {auc_score:.4f}")
print(f"Correct predictions: {np.sum(y_true == y_pred)}/{len(y_true)}")
print(f"Incorrect predictions: {np.sum(y_true != y_pred)}/{len(y_true)}")

#==============================================================================
# CONFUSION MATRIX
#==============================================================================

print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_true, y_pred)
print(f"\n{cm}\n")
print(f"True Negatives (Non-Recyclable correctly classified): {cm[0,0]}")
print(f"False Positives (Non-Recyclable misclassified as Recyclable): {cm[0,1]}")
print(f"False Negatives (Recyclable misclassified as Non-Recyclable): {cm[1,0]}")
print(f"True Positives (Recyclable correctly classified): {cm[1,1]}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300)  # ← Creates confusion_matrix.png
print(f"\nConfusion matrix saved to '{RESULTS_DIR}/confusion_matrix.png'")
plt.close()

#==============================================================================
# ROC CURVE
#==============================================================================

print("\n" + "="*70)
print("ROC CURVE")
print("="*70)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300)  # ← Creates roc_curve.png
print(f"ROC curve saved to '{RESULTS_DIR}/roc_curve.png'")
plt.close()

#==============================================================================
# SAMPLE PREDICTIONS
#==============================================================================

print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

# Get one batch of test images
test_generator.reset()
images, labels = next(test_generator)
images_display = images[:12]
labels_display = labels[:12]

# Predict
predictions = model.predict(images_display, verbose=0)
pred_labels = (predictions > 0.5).astype(int).flatten()

# Plot
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(12):
    axes[i].imshow(images_display[i])
    axes[i].axis('off')
    
    true_label = class_names[int(labels_display[i])]
    pred_label = class_names[pred_labels[i]]
    confidence = predictions[i][0] if pred_labels[i] == 1 else 1 - predictions[i][0]
    
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(
        f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}',
        color=color, fontweight='bold', fontsize=10
    )

plt.suptitle('Sample Predictions on Test Set', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sample_predictions.png'), dpi=300)  # ← Creates sample_predictions.png
print(f"Sample predictions saved to '{RESULTS_DIR}/sample_predictions.png'")
plt.close()

#==============================================================================
# FINAL SUMMARY
#==============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE - SUMMARY")
print("="*70)
print(f"\nDataset:")
print(f"  Training: {train_generator.samples} images")
print(f"  Validation: {val_generator.samples} images")
print(f"  Test: {test_generator.samples} images")
print(f"\nHyperparameters:")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Dense Units: {DENSE_UNITS}")
print(f"  Dropout Rate: {DROPOUT_RATE}")
print(f"  Phase 1 LR: {LEARNING_RATE_PHASE1}")
print(f"  Phase 2 LR: {LEARNING_RATE_PHASE2}")
print(f"  Unfrozen Layers: {UNFREEZE_LAYERS}")
print(f"\nFinal Test Results:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  AUC-ROC: {auc_score:.4f}")
print(f"\nFiles Saved:")
print(f"  Models:")
print(f"    - {MODELS_DIR}/model_phase1.h5")
print(f"    - {MODELS_DIR}/best_model.h5")
print(f"    - {MODELS_DIR}/final_model.h5")
print(f"  Visualizations:")
print(f"    - {RESULTS_DIR}/training_curves.png")
print(f"    - {RESULTS_DIR}/confusion_matrix.png")
print(f"    - {RESULTS_DIR}/roc_curve.png")
print(f"    - {RESULTS_DIR}/sample_predictions.png")
print("\n" + "="*70)
print("ALL DONE! Check the results folder for visualizations.")
print("="*70)