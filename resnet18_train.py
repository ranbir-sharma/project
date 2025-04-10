import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
import wandb
from dataset import *
from classification_evaluation import classifier
from tqdm import tqdm
from pprint import pprint
import argparse

# ---------------------- Classifier Evaluation ----------------------
def classifier(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            labels = torch.tensor([my_bidict[label] for label in labels], dtype=torch.long)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ---------------------- Training Function ----------------------
def train_or_test(model, data_loader, optimizer, criterion, device, args, epoch, mode='training'):
    model.train() if mode == 'training' else model.eval()
    running_loss = 0.0
    correct = 0

    for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader)):
        labels = torch.tensor([my_bidict[label] for label in labels], dtype=torch.long)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if mode == 'training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)

    if args.en_wandb:
        wandb.log({f"{mode}_loss": avg_loss, f"{mode}_acc": accuracy, "epoch": epoch})

    print(f"[{mode.upper()}] Epoch {epoch+1}: Loss = {avg_loss:.4f}, Acc = {accuracy:.4f}")
    return avg_loss, accuracy

# ---------------------- Main ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--en_wandb', type=bool, default=True)
    parser.add_argument('--tag', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='Resnet_models')
    parser.add_argument('--dataset', type=str, default='cpen455')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--load_params', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lr_decay', type=float, default=0.999995)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    pprint(args.__dict__)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.en_wandb:
        wandb.init(project="CPEN455-ResNet18", name=f"ResNet18_{args.tag}")
        wandb.config.update(args)

    ds_transforms = transforms.Compose([transforms.Resize((32, 32))])
    kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': True}

    train_loader = torch.utils.data.DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode='train', transform=ds_transforms),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode='validation', transform=ds_transforms),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # test_loader = torch.utils.data.DataLoader(
    #     CPEN455Dataset(root_dir=args.data_dir, mode='test', transform=ds_transforms),
    #     batch_size=args.batch_size, shuffle=False, **kwargs)

    model = models.resnet18(weights=None, num_classes=4)
    model = model.to(device)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params))
        print("Loaded model from checkpoint")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.max_epochs):
        train_or_test(model, train_loader, optimizer, criterion, device, args, epoch, mode='training')
        train_or_test(model, val_loader, optimizer, criterion, device, args, epoch, mode='validation')
        # train_or_test(model, test_loader, optimizer, criterion, device, args, epoch, mode='test')

        scheduler.step()

        if (epoch + 1) % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, f"resnet18_{args.dataset}_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

            # Log final accuracies to wandb
            if args.en_wandb:
                val_acc = classifier(model, val_loader, device)
                train_acc = classifier(model, train_loader, device)
                wandb.log({"Final Validation Accuracy": val_acc, "Final Train Accuracy": train_acc})
