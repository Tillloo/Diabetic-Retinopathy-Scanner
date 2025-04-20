import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
from dataExtraction import test_dataset
# from dataExtraction import test_dataset
import os

def plot_training_history(history_file):
    
    # Load training history
    history = pd.read_csv(history_file)
    
    # Create folder where resylts will be saved
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    # Save the accuracy plot 
    plt.savefig('results/accuracy.png')

    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Save the loss plot
    plt.savefig('results/loss.png')

def evaluate_model(test_dataset, model_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predict the test dataset
    y_true = []
    y_pred = []

    scores = model.predict(test_dataset,batch_size=32,verbose='auto',)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Parent folder
    DATA_DIR = os.path.join(BASE_DIR, 'data')  # 'data' folder
    TEST_DIR = os.path.join(DATA_DIR,'test')

    # Read labels
    LABELS_FILE = os.path.join(DATA_DIR, 'trainLabels.csv')  # Assuming the CSV is in 'data'
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

    labels = pd.read_csv(LABELS_FILE, dtype={"image":str, "level":np.int32})  # Read labels
    labels['image'] = labels['image'] + '.jpeg'  # Append '.jpeg' extension

    # Filter labels for images that exist
    fnames = os.listdir(TEST_DIR)
    mask = labels['image'].isin(fnames)
    test_labels = labels[mask].sort_values('image')

    score = pd.DataFrame(scores)
    score["level"] = np.argmax(score, axis=1)
    labels = ["0","1","2","3",'4']
    
    y_true = test_labels['level']
    y_pred = score["level"]
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig("Confusion_matrix.png")
    
    print("Precision score:",precision_score(y_true, y_pred,labels=labels,average=None))
    print("Recall Score score:",recall_score(y_true, y_pred,labels=labels,average=None))
    print("F1_score:",f1_score(y_true, y_pred,labels=labels,average=None))
    print("Accuracy Score:",accuracy_score(y_true, y_pred))


history = 'resNet_history.csv'
model = './resNet_50.keras'

# print("current path", os.getcwd())

# Plot training history
plot_training_history(history)
evaluate_model(test_dataset, model)
