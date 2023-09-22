import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Use GPU or TPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained generator
netG = Generator().to(device)
netG.load_state_dict(torch.load('generatorV38.pth'))
netG.eval()

fig = plt.figure(figsize=(20, 5))

# Generate and visualize 4 samples with different latent vectors
for i in range(4):
    # Generate a random latent vector z with 200 dimensions
    z = torch.randn((1, 200, 1, 1, 1)).to(device)
    #z = torch.randn((1, 200)).to(device)


    # Generate a 3D voxel data using the generator
    with torch.no_grad():
        generated_data = netG(z).cpu().numpy()

    # Reshape the data and visualize it using a 3D plot
    generated_data = generated_data.reshape((64, 64, 64))

    ax = fig.add_subplot(1, 4, i+1, projection='3d')

    # Find the coordinates of the voxels with values above the threshold (e.g., 0.3)
    coords = np.array(np.nonzero(generated_data > 0.3)).T

    # Get the intensity values for each coordinate (used for color)
    intensity = generated_data[generated_data > 0.3]

    # Create a scatter plot using the coordinates, with color determined by intensity and a colormap
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=intensity, cmap='viridis', marker='o', alpha=0.6)

    ax.set_title(f'Sample {i+1}')

plt.show()
