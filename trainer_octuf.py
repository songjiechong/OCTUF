import torch


def train(train_loader, model, criterion, sensing_rate, optimizer, device):
    model.train()
    sum_loss = 0
    for inputs in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    return sum_loss
