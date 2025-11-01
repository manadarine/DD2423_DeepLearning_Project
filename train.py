import torch
from utils import compute_loss

# training
def train(
    model, n_epochs, train_loader, val_loader,
    criterion, optimizer, lam=0.001, device='cuda'
):
    print("Starting training loop...")

    model.to(device)
    train_losses, train_costs, train_accuracies = [], [], []
    val_losses, val_costs, val_accuracies = [], [], []

    for epoch in range(n_epochs):
        print(f"Epoch [{epoch + 1}/{n_epochs}]")
        print("Epoch started, switching to training mode...") #added

        model.train()
        epoch_train_loss, epoch_train_cost, epoch_train_correct, epoch_train_total = 0, 0, 0, 0

        for inputs, labels, *_ in train_loader:
            print("Fetched a batch...")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            cost, loss = compute_loss(model, criterion, outputs, labels, lam)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * labels.size(0)
            epoch_train_cost += cost.item() * labels.size(0)
            epoch_train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            epoch_train_total += labels.size(0)

        # End of epoch training metrics
        avg_train_loss = epoch_train_loss / epoch_train_total
        avg_train_cost = epoch_train_cost / epoch_train_total
        avg_train_accuracy = epoch_train_correct / epoch_train_total

        train_losses.append(avg_train_loss)
        train_costs.append(avg_train_cost)
        train_accuracies.append(avg_train_accuracy)

        # Validation
        model.eval()
        val_loss, val_cost, val_correct, val_total = 0, 0, 0, 0

        with torch.no_grad():
            for inputs, labels, *_ in val_loader:
                
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                cost, loss = compute_loss(model, criterion, outputs, labels, lam)
                val_loss += loss.item() * labels.size(0)
                val_cost += cost.item() * labels.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_cost = val_cost / val_total
        avg_val_accuracy = val_correct / val_total

        val_losses.append(avg_val_loss)
        val_costs.append(avg_val_cost)
        val_accuracies.append(avg_val_accuracy)

        print(f"Epoch [{epoch + 1}/{n_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_accuracy:.4f}")

    return train_losses, train_costs, train_accuracies, val_losses, val_costs, val_accuracies
