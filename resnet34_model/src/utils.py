import os
import torch
import json
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(model, class_to_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(os.path.join(output_dir, "class_labels.json"), 'w') as f:
        json.dump(idx_to_class, f, indent=4)
        
def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy over Epochs')
    
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
