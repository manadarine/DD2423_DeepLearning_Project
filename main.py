import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data import load_all_batches, load_test_data
from model import VGG_3_layer
from utils import plot
from train import train
from test import test

def main(filename='regularization_only_3vgg', epochs=25, lam=0.001, lr=0.001):

    # seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)  # ADD THIS LINE


    # data
    train_images, train_labels, train_labels_onehot, val_images, val_labels, val_labels_onehot = load_all_batches()
    test_images, test_labels = load_test_data()

    # datasets
    # train_dataset = TensorDataset(train_images.reshape(-1, 3, 32, 32), train_labels.reshape(-1), train_labels_onehot.transpose(0, 1))
    train_dataset = TensorDataset(train_images, train_labels, train_labels_onehot.transpose(0, 1))
    val_dataset = TensorDataset(val_images, val_labels, val_labels_onehot.transpose(0, 1))
    test_dataset = TensorDataset(test_images, test_labels)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    # model
    model = VGG_3_layer()
    model.to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    train_losses, train_costs, train_accuracies, val_losses, val_costs, val_accuracies = train(model, epochs, train_loader, val_loader, criterion, optimizer, lam, device)
    test_loss, test_cost, test_accuracy = test(model, test_loader, criterion, lam, device)

    history = {
        'train_loss': train_losses,
        'train_cost': train_costs,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_cost': val_costs,
        'val_acc': val_accuracies
    }

    # create a folder for the results
    if not os.path.exists('results'):
        os.makedirs('results')
    plot(history, filename=f'results/{filename}.png')
    # store test results in a text file
    with open(f'results/{filename}.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Cost: {test_cost:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main(filename='regularization_only_3vgg', epochs=40, lam=0.001, lr=0.001)







