import matplotlib.pyplot as plt
import torch

# loss function
def compute_loss(model, criterion, outputs, labels, lam):
    ce_loss = criterion(outputs, labels)
    l2_reg = sum(torch.sum(param ** 2) for param in model.parameters() if param.requires_grad)
    total_loss = ce_loss + lam * l2_reg
    return total_loss, ce_loss

def plot(history, filename='accuracy_loss_cost.png'):

    train_accs = history['train_acc']
    val_accs = history['val_acc']
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    train_costs = history['train_cost']
    val_costs = history['val_cost']

    plt.figure(figsize=(15, 5))

    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Cost
    plt.subplot(1, 3, 3)
    plt.plot(train_costs, label='Train Cost')
    plt.plot(val_costs, label='Validation Cost')
    plt.title('Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
