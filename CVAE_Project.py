import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision                                                                                                                             
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)                                                      # Define a simple CVAE class with MLP architecture

class CVAE_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super(CVAE_MLP, self).__init__()

        # Encoder: input_dim + num_classes -> hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # This should output hidden_dim, not latent_dim * 2
        )
        
        # Separate layers for mu and logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Classification head to predict class from latent space
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        # Decoder: latent_dim + num_classes -> hidden_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output should be in range [0, 1] for MNIST
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Random normal noise
        return mu + eps * std

    def forward(self, x, y):
        # Flatten the input images and one-hot labels
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        
        # Concatenate image data with labels and pass through encoder
        x = torch.cat((x, y), dim=-1)
        hidden = self.encoder(x)
        
        # Compute mu and logvar
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Classification logits from the latent space
        class_logits = self.classifier(z)
        
        # Concatenate latent vector with labels and pass through decoder
        z = torch.cat((z, y), dim=-1)
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar, class_logits

def cvae_loss(recon, data, mu, logvar, class_logits, labels):
    # Flatten the data tensor
    data = data.view(data.size(0), -1)
    
    # Reconstruction loss (binary cross-entropy)
    reconstruction_loss = F.binary_cross_entropy(recon, data, reduction='sum')
    
    # KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Cross-entropy loss for class prediction
    classification_loss = F.cross_entropy(class_logits, labels)
    
    # Total loss
    return reconstruction_loss + kl_divergence + classification_loss
def train_cvae_mlp(model, train_loader, num_epochs=10, learning_rate=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')  # Initialize with a high value
    best_model = None

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # One-hot encode the labels
            labels_one_hot = F.one_hot(labels, num_classes=10).float()

            # Forward pass through the model
            recon, mu, logvar, class_logits = model(data, labels_one_hot)
            
            # Calculate the loss
            loss = cvae_loss(recon, data, mu, logvar, class_logits, labels)
            
            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f'CVAE-MLP Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}')

        # Visualize some generated images after each epoch
        if (epoch + 1) % 1 == 0:
            print("Sample Images:")
            with torch.no_grad():
                num_classes = 10  # Number of classes (0 to 9)
                z = torch.randn(num_classes, model.fc_mu.out_features)  # Latent space samples
                y = torch.eye(num_classes)  # One-hot encoded class labels
                sample = torch.cat([z, y], dim=1)
                sample = model.decoder(sample).view(num_classes, 1, 28, 28)
                sample = sample.squeeze().cpu()
                fig, axs = plt.subplots(1, num_classes, figsize=(15, 2))
                for i in range(num_classes):
                    axs[i].imshow(sample[i], cmap='gray')
                    axs[i].set_title(f"Class {i}", fontsize=16)
                    axs[i].axis('off')
                plt.show()

        # Save the best model based on loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model.state_dict()

    # Save the best model to a file
    torch.save(best_model, 'best_cvae_mlp_model.pth')
    print("Best model saved as 'best_cvae_mlp_model.pth'")
# Parameters for MNIST
input_dim = 28 * 28  # Flattened image size for MNIST (28x28 pixels)
hidden_dim = 256  # Number of hidden units in the MLP
latent_dim = 2  # Latent space dimension (2 for visualization)
num_classes = 10  # Number of classes in MNIST (digits 0-9)

# Create an instance of the CVAE_MLP model
cvae_mlp = CVAE_MLP(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_classes=num_classes)
def visualize_latent_space(model, data_loader, num_samples=1000):
    model.eval()  # Set the model to evaluation mode
    z_list = []
    labels_list = []

    # We will collect latent vectors (mu) and their corresponding labels
    with torch.no_grad():
        for i, (data, labels) in enumerate(data_loader):
            if len(z_list) * data.size(0) > num_samples:
                break  # Stop once we have enough samples

            # One-hot encode the labels
            labels_one_hot = F.one_hot(labels, num_classes=10).float()

            # Forward pass to get the latent variables
            _, mu, _, _ = model(data, labels_one_hot)
            
            z_list.append(mu)
            labels_list.append(labels)
        
        # Concatenate all the latent vectors and labels
        z = torch.cat(z_list).cpu().numpy()
        labels = torch.cat(labels_list).cpu().numpy()
        
        # Plot latent space with different colors for different classes
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Latent Space Distribution (CVAE MLP)')
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.show()


train_cvae_mlp(cvae_mlp, train_loader, num_epochs=10, learning_rate=1e-3)
visualize_latent_space(cvae_mlp, train_loader)

