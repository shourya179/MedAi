import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# paths
TRAIN_DIR = 'data/chest_xray/train'
TEST_DIR  = 'data/chest_xray/test'

# image transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),      # resize all images to same size
    transforms.RandomHorizontalFlip(),  # data augmentation
    transforms.RandomRotation(10),      # slight rotation
    transforms.ToTensor(),              # convert to tensor
    transforms.Normalize(               # normalize pixels
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# load datasets
train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_data  = datasets.ImageFolder(TEST_DIR,  transform=test_transforms)

# data loaders — load images in batches
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

print(f"Classes: {train_data.classes}")
print(f"Training images: {len(train_data)}")
print(f"Test images: {len(test_data)}")

# use pretrained ResNet18 — transfer learning!
model = models.resnet18(weights='IMAGENET1K_V1')

# freeze all layers except the last one
for param in model.parameters():
    param.requires_grad = False

# replace final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)

# class weights for imbalanced data
class_weights = torch.tensor([3875/1341, 1.0])  # weight normal class higher
loss_fn       = nn.CrossEntropyLoss(weight=class_weights)
optimizer     = optim.Adam(model.fc.parameters(), lr=0.001)

# training loop
print("\nTraining CNN...")
for epoch in range(5):
    model.train()
    running_loss = 0
    correct      = 0
    total        = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted     = outputs.argmax(1)
        correct      += (predicted == labels).sum().item()
        total        += labels.size(0)

    train_acc = correct / total

    # evaluate on test
    model.eval()
    test_correct = 0
    test_total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs   = model(images)
            predicted = outputs.argmax(1)
            test_correct += (predicted == labels).sum().item()
            test_total   += labels.size(0)

    test_acc = test_correct / test_total
    print(f"Epoch:{epoch+1} - Loss:{running_loss/len(train_loader):.4f} - Train:{train_acc:.4f} - Test:{test_acc:.4f}")

# save model
os.makedirs('models/saved', exist_ok=True)
torch.save(model.state_dict(), 'models/saved/xray_model.pth')
print("\nModel saved! ✅")