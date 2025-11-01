import torch
from utils import compute_loss

# testing
def test(model, test_loader, criterion, lam=0.001, device='cuda'):
    model.eval()
    total_loss = []
    total_cost = []
    total_test_accuracy = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            cost, loss = compute_loss(model, criterion, outputs, labels, lam)
            accuracy = (outputs.argmax(dim=1) == labels).sum().item() / labels.size(0)
            total_loss.append(loss.item())
            total_cost.append(cost.item())
            total_test_accuracy.append(accuracy)

    avg_loss = sum(total_loss) / len(total_loss)
    avg_cost = sum(total_cost) / len(total_cost)
    avg_test_accuracy = sum(total_test_accuracy) / len(total_test_accuracy)

    return avg_loss, avg_cost, avg_test_accuracy