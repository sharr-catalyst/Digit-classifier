import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # set GPU Memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration warning: {e}")
    
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"Error importing TensorFlow: {e}")
    exit(1)
# Load the dataset
print("\nLoading dataset...")
try:
    data = pd.read_csv('mnist_train.csv')
    print(f"Dataset loaded successfully: {len(data)} samples")
except FileNotFoundError:
    print("ERROR: train.csv not found!")
    print("Please download it from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
    exit(1)

# Separate features and labels
y = data['label'].values
X = data.drop('label', axis=1).values
X = X.astype('float32') / 255.0
X = X.reshape(-1, 28, 28, 1)

print(f"Input shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Build the model
print("\nBuilding model...")
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
], name='MNIST_Classifier')

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Train the model
print("\nTraining model...")
try:
   history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_data=(X_val, y_val),
        verbose=1
    )
except Exception as e:
    print(f"Training error: {e}")
    print("\nTrying with smaller batch size...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {test_loss:.4f}")
print(f"Validation Accuracy: {test_accuracy * 100:.2f}%")

# Plot training history
print("\nGenerating training plots...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("Training history plot saved as 'training_history.png'")

# Make predictions on sample images
print("\nMaking predictions on sample images...")
sample_indices = np.random.choice(len(X_val), 10, replace=False)
sample_images = X_val[sample_indices]
sample_labels = y_val[sample_indices]

predictions = model.predict(sample_images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
confidence_scores = np.max(predictions, axis=1)

# Display sample predictions
plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    color = 'green' if sample_labels[i] == predicted_labels[i] else 'red'
    plt.title(f"True: {sample_labels[i]} | Pred: {predicted_labels[i]}\n"
              f"Confidence: {confidence_scores[i]:.2%}", 
              color=color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
print("Sample predictions saved as 'sample_predictions.png'")

# Save the model
print("\nSaving model...")
try:
    model.save('mnist_digit_classifier.keras')  # Use .keras format (recommended)
    print("Model saved as 'mnist_digit_classifier.keras'")
except:
    model.save('mnist_digit_classifier.h5')  # Fallback to .h5 format
    print("Model saved as 'mnist_digit_classifier.h5'")
# Print final summary
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Final Validation Accuracy: {test_accuracy * 100:.2f}%")
print(f"Final Validation Loss: {test_loss:.4f}")
print(f"Total Parameters: {model.count_params():,}")
print("="*50)

# Show plots
plt.show()
#TEST MODEL
print("\n" + "="*50)
print("TESTING MODEL WITH TEST DATASET")
print("="*50)

try:
    print("\nLoading test dataset...")
    print("Download test.csv from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
    
    test_data = pd.read_csv('mnist_test.csv')
    print(f"Test dataset loaded: {len(test_data)} samples")
    
    # Check if test data has labels or not
    if 'label' in test_data.columns:
        # Test data with labels
        y_test = test_data['label'].values
      X_test = test_data.drop('label', axis=1).values
        has_labels = True
    else:
        # Test data without labels (competition format)
        X_test = test_data.values
        has_labels = False
    
    # Preprocess test data
    X_test = X_test.astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"Test data shape: {X_test.shape}")
    
    # Make predictions
    print("\nMaking predictions on test data...")
    test_predictions = model.predict(X_test, verbose=1)
    predicted_test_labels = np.argmax(test_predictions, axis=1)
    
    if has_labels:
        test_acc = np.mean(predicted_test_labels == y_test)
        print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
        
        #confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
      cm = confusion_matrix(y_test, predicted_test_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix - Test Data')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        print("Confusion matrix saved as 'confusion_matrix.png'")
        
        print("\nClassification Report:")
        print(classification_report(y_test, predicted_test_labels, 
                                    target_names=[str(i) for i in range(10)]))
        
        # Visualize some test predictions
        plt.figure(figsize=(15, 6))
        sample_test_indices = np.random.choice(len(X_test), 15, replace=False)
        
        for i, idx in enumerate(sample_test_indices):
            plt.subplot(3, 5, i + 1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            color = 'green' if y_test[idx] == predicted_test_labels[idx] else 'red'
            plt.title(f"True: {y_test[idx]} | Pred: {predicted_test_labels[idx]}", 
                     color=color, fontsize=8)
            plt.axis('off')
          plt.suptitle('Test Set Predictions', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('test_predictions.png', dpi=150)
        print("Test predictions saved as 'test_predictions.png'")
        plt.show()
    else:
        # Save predictions to CSV (competition format)
        print("\nNo labels found in test data. Saving predictions to CSV...")
        submission = pd.DataFrame({
            'ImageId': range(1, len(predicted_test_labels) + 1),
            'Label': predicted_test_labels
        })
        submission.to_csv('predictions.csv', index=False)
        print("Predictions saved as 'predictions.csv'")
        
        # Visualize random test predictions
        plt.figure(figsize=(15, 6))
        sample_test_indices = np.random.choice(len(X_test), 15, replace=False)
        
        for i, idx in enumerate(sample_test_indices):
            plt.subplot(3, 5, i + 1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            confidence = test_predictions[idx][predicted_test_labels[idx]]
            plt.title(f"Pred: {predicted_test_labels[idx]}\n"
                     f"Conf: {confidence:.2%}", fontsize=8)
            plt.axis('off')
        
        plt.suptitle('Test Set Predictions (No Labels)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('test_predictions.png', dpi=150)
        print("Test predictions saved as 'test_predictions.png'")
        plt.show()
    
    print("\n" + "="*50)
    print("TESTING COMPLETE")
    print("="*50)
    
except FileNotFoundError:
    print("\ntest.csv not found!")
    print("Download it from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
    print("Place it in the same directory as this script.")
except Exception as e:
    print(f"\nError during testing: {e}")
    print("Make sure test.csv is properly formatted.")

