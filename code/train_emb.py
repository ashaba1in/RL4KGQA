import torch
from torch import nn
import numpy as np


def train_emb(model, dataset, optimizer, criterion, batch_size):
    device = next(model.parameters()).device

    model.train()

    epoch_loss = []
    accuracy = []
    all_ranks = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        entities, relations, labels = zip(*batch)

        entities = torch.tensor(entities, device=device)
        relations = torch.tensor(relations, device=device)
        labels = torch.tensor(labels, device=device)

        optimizer.zero_grad()

        output = model(entities, relations)

        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        ranks = (output >= output.gather(1, labels.view(-1, 1))).sum(1)
        all_ranks.extend(ranks.detach().cpu())
        accuracy.append((output.argmax(axis=1) == labels).detach().cpu().float().mean())
        epoch_loss.append(loss.item())

    return np.mean(epoch_loss), np.mean(accuracy), all_ranks


@torch.no_grad()
def evaluate_emb(model, dataset, criterion, batch_size):
    device = next(model.parameters()).device

    model.eval()

    epoch_loss = []
    accuracy = []
    all_ranks = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        entities, relations, labels = zip(*batch)

        entities = torch.tensor(entities, device=device)
        relations = torch.tensor(relations, device=device)
        labels = torch.tensor(labels, device=device)

        output = model(entities, relations)

        loss = criterion(output, labels)

        ranks = (output >= output.gather(1, labels.view(-1, 1))).sum(1)
        all_ranks.extend(ranks.detach().cpu())
        accuracy.extend((output.argmax(axis=1) == labels).detach().cpu().float())
        epoch_loss.append(loss.item())

    return np.mean(epoch_loss), np.mean(accuracy), all_ranks