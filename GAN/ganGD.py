import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load("modelnet10.npz", allow_pickle=True)
train_voxel = data["train_voxel"]

# Convert data to PyTorch tensors
train_voxel = torch.tensor(train_voxel, dtype=torch.float32).unsqueeze(1)

# Create data loaders
dataset = TensorDataset(train_voxel)
train_loader = DataLoader(dataset, batch_size=40, shuffle=True)


# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(200, 512, 4, 1, 0, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.ConvTranspose3d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Define the Discriminator
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

# Create the models, define loss function and optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0025, betas=(0.5, 0.999))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters in Generator: {count_parameters(netG)}')
print(f'Total parameters in Discriminator: {count_parameters(netD)}')

G_losses = []
D_losses = []

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):

        # Update Discriminator
        netD.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), 1., device=device)
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(batch_size, 200, 1, 1, 1, device=device)
        fake_data = netG(noise)
        label.fill_(0.)
        output = netD(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()

        # Calculate discriminator accuracy
        pred_real = netD(real_data).view(-1)
        pred_fake = netD(fake_data.detach()).view(-1)
        acc_real = ((pred_real > 0.5).float() == label).float().mean()
        label.fill_(1.)
        acc_fake = ((pred_fake < 0.5).float() == label).float().mean()
        disc_acc = (acc_real + acc_fake) / 2

        # Update discriminator only if accuracy is below 0.8
        if disc_acc < 0.8:
            optimizerD.step()

        # (2) Update Generator
        netG.zero_grad()
        label.fill_(1.)
        output = netD(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD_real.item() + errD_fake.item())

        # Print statistics
        print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] Loss_D: {errD_real.item()+errD_fake.item()} Loss_G: {errG.item()}')

# Save models
torch.save(netG.state_dict(), 'generatorV38.pth')
torch.save(netD.state_dict(), 'discriminatorV38.pth')

# Plot the trainging loss
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
