import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Load data and labels
data = np.load("modelnet10.npz", allow_pickle=True)
train_voxel = data["train_voxel"]
train_labels = data["train_labels"] # Make sure to load the correct label data

# Convert data to PyTorch tensors
train_voxel = torch.tensor(train_voxel, dtype=torch.float32).unsqueeze(1)
train_labels = torch.tensor(train_labels, dtype=torch.long) # Labels as long tensor

# Create data loaders
dataset = TensorDataset(train_voxel, train_labels)
train_loader = DataLoader(dataset, batch_size=40, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def feature_maps(self, input):
        features = []
        x = input
        for layer in self.main.children():
            x = layer(x)
            features.append(x)
        return features[1], features[3], features[5]

# Define the Classifier
class Classifier(nn.Module):
    def __init__(self, discriminator, num_classes):
        super(Classifier, self).__init__()
        self.discriminator = discriminator
        self.fc = nn.Linear(28672, num_classes)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        fmap2, fmap3, fmap4 = self.discriminator.feature_maps(x)
        fmap2 = nn.MaxPool3d(8)(fmap2)
        fmap3 = nn.MaxPool3d(4)(fmap3)
        fmap4 = nn.MaxPool3d(2)(fmap4)
        fmap = torch.cat((fmap2, fmap3, fmap4), 1)
        fmap = fmap.view(fmap.size(0), -1)
        x = self.fc(fmap)
        #x = self.softmax(x)
        return x

# Load the trained discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netD = Discriminator().to(device)
netD.load_state_dict(torch.load('discriminatorV38.pth'))

# Define number of classes (replace with your actual number of classes)
num_classes = 10
classifier = Classifier(netD, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Training loop
num_epochs = 5 # or more
training_losses = []
training_accuracies = []

for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        epoch_loss += loss.item()

    training_losses.append(epoch_loss / (i+1))
    training_accuracies.append(100 * correct / total)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss / (i+1)}, Accuracy: {100 * correct / total}%')

# Plot the training loss and accuracy
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(training_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()
