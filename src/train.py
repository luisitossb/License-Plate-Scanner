import torch
from src.model import compute_iou


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_iou = 0.0
    for images, bboxes in loader:
        images, bboxes = images.to(device), bboxes.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, bboxes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
        total_iou  += compute_iou(preds.detach(), bboxes).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_iou / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_iou = 0.0
    for images, bboxes in loader:
        images, bboxes = images.to(device), bboxes.to(device)
        preds = model(images)
        total_loss += criterion(preds, bboxes).item() * len(images)
        total_iou  += compute_iou(preds, bboxes).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_iou / n
