import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions

from torch.utils.data import TensorDataset, DataLoader

import numpy as np


### Checking GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"RUNNING ON DEVICE: {device}")


### DATA LOADING

def get_dataloader(filepath, batch_size):
    np_vision_frames = np.load(filepath)
    tensor_vision_frames = torch.Tensor(np_vision_frames)

    vision_dataset = TensorDataset(tensor_vision_frames)
    vision_dataloader = DataLoader(vision_dataset, batch_size=batch_size, shuffle=True)

    return vision_dataloader





### NETWORK DEFINITIONS - Convolutional VAE

class VariationalEncoder(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(VariationalEncoder, self).__init__()

        self.flatten = nn.Flatten()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
        )


        # compute shape with one forward pass
        with torch.no_grad():
            observation_shape = (1, 80, 120) # generalise so I can pass as input arguments
            n_flatten = self.cnn(torch.zeros(observation_shape)[None]).shape
            print(n_flatten)


        self.dense_seq = nn.Sequential(
            nn.Linear(n_flatten[1]*n_flatten[2]*n_flatten[3], 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )


        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_sigma = nn.Linear(512, latent_dim)



        # distributions
        self.N = distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # GPU
        self.N.scale = self.N.scale.cuda()

        self.KL = 0

    def forward(self, x):
        # x = self.variational_encoder(x)
        x = self.cnn(x)

        # print(f"Encoded shape: {x.shape}")

        x = self.flatten(x)

        x = self.dense_seq(x)

        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))

        z = mu + sigma * self.N.sample(mu.shape) # reparameterisation trick
        self.KL = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z


class Decoder(nn.Module):
    def __init__(self, image_channels, unflat_shape, latent_dim):
        super(Decoder, self).__init__()

        self.unflat_shape = unflat_shape

        print(f"RESULT: {self.unflat_shape[1] * self.unflat_shape[2] * self.unflat_shape[3]}")

        self.decoder_dense_seq = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.unflat_shape[1] * self.unflat_shape[2] * self.unflat_shape[3]),
            nn.ReLU()
        )

        self.transpose_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=image_channels, kernel_size=3, stride=2),
            nn.Sigmoid()
        )


    def forward(self, x):
        # for final padding
        original_shape = x.shape

        x = self.decoder_dense_seq(x)

        # reshape
        x = x.view(-1, self.unflat_shape[1], self.unflat_shape[2], self.unflat_shape[3])

        x = self.transpose_cnn(x)

        dummy_row = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3]).to(device)
        x = torch.cat((x, dummy_row), 2)

        dummy_col = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1).to(device)
        x = torch.cat((x, dummy_col), 3)

        return x

class VAE(nn.Module):
    def __init__(self, image_channels=1, latent_dim=256):
        super().__init__()

        self.encoder = VariationalEncoder(image_channels, latent_dim)

        unflat_shape = 0
        with torch.no_grad():            
            observation_shape = (1, 80, 120) # generalise so I can pass as input arguments
            unflat_shape = self.encoder.cnn(torch.ones(observation_shape)[None]).shape


        self.decoder = Decoder(image_channels, unflat_shape, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


### TRAINING METHODS

def train(vae: VAE, dataloader: DataLoader, optimizer, batch_size):
    size = len(dataloader.dataset)

    vae.train()

    for count, X in enumerate(dataloader):
        X = X[0] # weird list formatting in dataloader TODO fix
        X = X.view(batch_size, 1, 80, 120)

        X = X.to(device)

        vae_pred = vae(X)

        loss_fn = nn.MSELoss().to(device)

        # loss = ((X - vae_pred)**2).sum() + vae.encoder.KL
        loss = loss_fn(vae_pred, X) + vae.encoder.KL # this loss function may not work very well

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if(not(count % 1600)):
            loss = loss.item()
            current = count * batch_size + len(X)
            print(f"Loss: {loss:>5f} [{current:>5d}/{size:>5d}]")


def test(vae: VAE, dataloader: DataLoader):
    num_batches = len(dataloader)

    vae.eval()

    test_loss = 0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)

            vae_pred = vae(X)
            test_loss += ((X - vae_pred)**2).sum() + vae.encoder.KL

    test_loss /= num_batches

    print(f"\nAverage loss: {test_loss:>5f}")


### MAIN FUNCTIONS

def main():

    # hyperparameters
    learning_rate = 1e-3
    batch_size = 4
    epochs = 100

    # loading data
    filepath = 'mm_testing/dataset/vision_frames.npy'
    training_dataloader = get_dataloader(filepath=filepath, batch_size=batch_size) 

    # instantiating model
    variational_autoencoder = VAE().to(device)

    # model attributes
    optimizer = torch.optim.Adam(variational_autoencoder.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------")
        train(variational_autoencoder, training_dataloader, optimizer, batch_size)
        # test(variational_autoencoder, testing_dataloader)
        print("\n\n")

        if not(t % (epochs // 4)):
            torch.save(variational_autoencoder.state_dict(), 'VAE_checkpoint.pth')

    torch.save(variational_autoencoder.state_dict(), 'VAE.pth')

    print("Job Finished.")

if __name__ == "__main__":
    main()