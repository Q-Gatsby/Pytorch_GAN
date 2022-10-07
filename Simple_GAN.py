import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# Things to try
# 1.What happens if you use larger network?
# 2.Better normalization with BatchNorm
# 3.Different learning rate?
# 4.Change architecture to CNN?


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 3e-4
z_dim = 64
img_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

discriminator = Discriminator(in_features=img_dim).to(device)
generator = Generator(z_dim=z_dim, img_dim=img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose([
    # torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))
    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=0.5, std=0.5)
])
# 上面的ToTensor()可以完成归一化，同时Normalize把mean和std设置为0.5,可以把所有[0,1]的值投影到[-1,1]的区间内
# 这也是为什么最后的Generator要加一层tanh，是因为tanh会把所有的值投射到[-1,1]

datasets = datasets.MNIST(root='./MNIST_dataset', transform=transforms, download=True)
dataloader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True)
optimizer_disc = torch.optim.Adam(params=discriminator.parameters(), lr=learning_rate)
optimizer_gen = torch.optim.Adam(params=generator.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

for epoch in range(num_epochs):
    for batch_idx, (imgs, _) in enumerate(dataloader):
        imgs = imgs.reshape(-1, 784).to(device)
        batch_size = imgs.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # 但通过查看BCELoss的Pytorch官方文档，可以看到其表达式前面存在一个负号，因而对于lossD来说是要Minimize loss
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = generator(noise)
        disc_real = discriminator(imgs).reshape(-1)
        lossD_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake).reshape(-1)
        lossD_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        optimizer_disc.zero_grad()
        lossD.backward(retain_graph=True)  # retain the grads and the grads will not be freed
        optimizer_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = discriminator(fake).reshape(-1)
        lossG = loss_fn(output, torch.ones_like(output))
        optimizer_gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                              Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                data = imgs.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
