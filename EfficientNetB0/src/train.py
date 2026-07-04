import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import OUTPUT_DIR, STAGE_A_LR, STAGE_A_EPOCHS, STAGE_B_EPOCHS, EARLY_STOPPING_PATIENCE
from dataset import get_dataloaders
from model import build_model, classifier_parameters, get_stage_b_params
from utils import EarlyStopping, save_checkpoint, plot_metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders()
    num_classes = len(class_to_idx)
    print(f"Detected {num_classes} classes.")

    print("Initializing model in its standard configuration...")
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print("\n--- STAGE A: Training classifier head (backbone frozen) ---")
    for p in model.parameters():
        p.requires_grad = False
    for p in classifier_parameters(model):
        p.requires_grad = True
    optimizer_A = optim.Adam(classifier_parameters(model), lr=STAGE_A_LR)

    for epoch in range(STAGE_A_EPOCHS):
        t0 = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer_A, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc); val_accs.append(v_acc)
        print(f"Stage A Epoch {epoch+1}/{STAGE_A_EPOCHS} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} | {time.time()-t0:.1f}s")

    print("\n--- STAGE B: Standard partial-unfreeze fine-tune ---")
    stage_b_groups = get_stage_b_params(model)
    optimizer_B = optim.Adam(stage_b_groups)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    best_val_loss = float('inf')

    for epoch in range(STAGE_B_EPOCHS):
        t0 = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer_B, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc); val_accs.append(v_acc)
        print(f"Stage B Epoch {epoch+1}/{STAGE_B_EPOCHS} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} | {time.time()-t0:.1f}s")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            print("  -> Validation loss decreased. Saving best model...")
            save_checkpoint(model, class_to_idx, OUTPUT_DIR)

        early_stopping(v_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after {epoch+1} epochs in Stage B.")
            break

    print("\nTraining completed! Generating plots...")
    plot_metrics(train_losses, val_losses, train_accs, val_accs, OUTPUT_DIR)
    print(f"Plots and best model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
