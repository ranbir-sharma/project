import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from dataset import *
from bidict import bidict
import torch.nn as nn
import csv


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "Resnet_models/resnet18_cpen455_199.pth"  # change if needed
batch_size = 64

# Transforms
transform = transforms.Compose([
    transforms.Resize((32, 32))
])

# Dataloaders
train_loader = DataLoader(CPEN455Dataset("data", mode='train', transform=transform),
                          batch_size=batch_size, shuffle=False)
val_loader = DataLoader(CPEN455Dataset("data", mode='validation', transform=transform),
                        batch_size=batch_size, shuffle=False)
test_loader = DataLoader(CPEN455Dataset("data", mode='test', transform=transform),
                        batch_size=batch_size, shuffle=False)

# Load model
model = models.resnet18(weights=None, num_classes=4)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Evaluation function
def evaluate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = torch.tensor([my_bidict[label] for label in labels], dtype=torch.long)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def evaluate_Test(model, loader):
    input_csv_path="data/test.csv"
    output_csv_path="data/test1.csv"
    with torch.no_grad():
        predictions = []
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            # print(f"Preds {preds}")
            for pred in preds:
                predictions.append(int(pred))
        
        with open(input_csv_path, mode="r") as infile, open(output_csv_path, mode="w", newline="") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            index = 0

            for row in reader:
                img_path = row[0]

                row[1] = int(predictions[index])
                index = index + 1
                writer.writerow(row)


# Compute accuracies
train_acc = evaluate(model, train_loader)
val_acc = evaluate(model, val_loader)
evaluate_Test(model, test_loader)

print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
