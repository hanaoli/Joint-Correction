import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.autograd import Variable
import fid_score

def plot_energy(ET, EF):
    x = range(len(ET))
    plt.figure()
    plt.plot(x, ET, color='red', label='Data')
    plt.plot(x, EF, color='blue', label='Noise')
    plt.legend(loc="upper left")
    plt.savefig(folder + "/energy.png")
    plt.close()

def plot_energy2(ET, EF):
    x = range(len(ET[1:]))
    plt.figure()
    plt.plot(x, ET[1:], color='red', label='Data')
    plt.plot(x, EF[1:], color='blue', label='Noise')
    plt.legend(loc="upper left")
    plt.savefig(folder + "/energy2.png")
    plt.close()

def plot_acc(true_acc, false_acc):
    x = range(len(true_acc))
    plt.figure()
    plt.plot(x, true_acc, color='red', label='True Accuracy')
    plt.plot(x, false_acc, color='blue', label='False Accuracy')
    plt.legend(loc="upper left")
    plt.savefig(folder + "/accuracy.png")
    plt.close()

def plot_loss_nce(loss):
    x = range(len(loss))
    plt.figure()
    plt.plot(x, loss, color='red', label='Loss')
    plt.legend(loc="upper left")
    plt.savefig(folder + "/loss.png")
    plt.close()

def plot_recon(recon):
    x = range(len(recon))
    plt.figure()
    plt.plot(x, recon, color='red', label='Recon')
    plt.legend(loc="upper left")
    plt.savefig(folder + "/recon.png")
    plt.close()




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
    #        nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim * img_dim * channel),
            nn.Tanh()
        )

    def forward(self, z):
        return self.layers(z)


class Generator_CelebA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0)
       # self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
      #  self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
       # self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.2)
    # bn
    def forward(self, z):
        x = self.relu(self.conv1(z.view(-1, latent_dim, 1, 1)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.sigmoid(self.conv5(x))
        return x.view(-1, channel*img_dim*img_dim)


class Generator_SVHN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0)
       # self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
      #  self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
       # self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(0.2)
    # bn
    def forward(self, z):
        x = self.relu(self.conv1(z.view(-1, latent_dim, 1, 1)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.tanh(self.conv4(x))
        return x.view(-1, channel*img_dim*img_dim)

class Classifier_SVHN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(img_dim * img_dim * channel + latent_dim, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(latent_dim * 2, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)
        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, latent_dim, kernel_size=4, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, z):
        x = self.lrelu(self.conv1(x.view(-1, channel, img_dim, img_dim)))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        xz = torch.cat([x.view(-1, latent_dim), z], dim=1)
        p = self.relu(self.fc1(xz))
        p = self.relu(self.fc2(p))
        p = self.sigmoid(self.fc3(p))
        return p

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(img_dim * img_dim * channel + latent_dim, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(latent_dim * 2, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)
        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, latent_dim, kernel_size=4, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, z):
        x = self.lrelu(self.conv1(x.view(-1, channel, img_dim, img_dim)))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        xz = torch.cat([x.view(-1, latent_dim), z], dim=1)
        p = self.relu(self.fc1(xz))
        p = self.relu(self.fc2(p))
        p = self.sigmoid(self.fc3(p))
        return p


def init_weights(Layer):
    name = Layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(Layer.weight, mean=0, std=0.02)
        if Layer.bias is not None:
            torch.nn.init.constant_(Layer.bias, 0)


class Classifier_CelebA(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(img_dim * img_dim * channel + latent_dim, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(latent_dim * 2, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)
        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, latent_dim, kernel_size=4, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, z):
        x = self.lrelu(self.conv1(x.view(-1, channel, img_dim, img_dim)))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.conv5(x)
        xz = torch.cat([x.view(-1, latent_dim), z], dim=1)
        p = self.relu(self.fc1(xz))
        p = self.relu(self.fc2(p))
        p = self.sigmoid(self.fc3(p))
        return p

def init_weights(Layer):
    name = Layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(Layer.weight, mean=0, std=0.02)
        if Layer.bias is not None:
            torch.nn.init.constant_(Layer.bias, 0)

folder = "results_correction_scratch_celeba_tanh"
if not os.path.exists(folder):
    os.makedirs(folder)



batch_size = 100
n_epochs = 100
load_model = False
dataset = 'celeba'
n_show = 10
step_size = 0.1
steps = 30
crop = lambda x: transforms.functional.crop(x, 45, 25, 173 - 45, 153 - 25)
transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
transform_celeba = transforms.Compose([transforms.Lambda(crop), transforms.Resize(64),transforms.ToTensor()])
#, transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
if dataset == 'mnist':
    train_data = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True,
                                           transform=transform_mnist),
                batch_size=batch_size, shuffle=False)
    img_dim = 28
    channel = 1
    latent_dim = 30
    if load_model:
        G = torch.load("vae/g.pth")
    else:
        G = Generator().cuda()
        D = Classifier().cuda()

elif dataset == 'svhn':
    train_data = torch.utils.data.DataLoader(
                torchvision.datasets.SVHN(root='./data/SVHN', split='train', download=True,
                                           transform=transform),
                batch_size=batch_size, shuffle=False)
    img_dim = 32
    channel = 3
    latent_dim = 100
    if load_model:
        G = torch.load("vae/g_svhn.pth")
    else:
        G = Generator_SVHN().cuda()
        D = Classifier_SVHN().cuda()


elif dataset == 'celeba':
    train_data = torch.utils.data.DataLoader(
                torchvision.datasets.CelebA(root='./data/CelebA', split='train', download=True,
                                           transform=transform_celeba),
                batch_size=batch_size, shuffle=False)
    img_dim = 64
    channel = 3
    latent_dim = 100
    if load_model:
        G = torch.load("vae/g_celeba.pth")
    else:
        G = Generator_CelebA().cuda()
        D = Classifier_CelebA().cuda()

def langevin_dynamics_generator(z, obs, training=True):
    obs = obs.detach()
    criterian = nn.MSELoss(reduction='sum')
    z = Variable(z, requires_grad=True)
    for i in range(steps):
        noise = Variable(torch.randn(z.shape[0], latent_dim).cuda())
        gen_res = G(z)

        loss = 1.0 / (2.0 * 0.3 * 0.3) * criterian(gen_res, obs) - (torch.log(D(gen_res, z) + 1e-6) - torch.log(1 - D(gen_res, z) + 1e-6)).sum()
        grad = torch.autograd.grad(loss, z)[0]
        z.data = z.data - 0.5 * step_size * step_size * (z.data + grad)
        if training:
            z.data += step_size * noise
    return z

optimizer_D = torch.optim.Adam(D.parameters(), lr=0.00005, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))

#optimizers
#optimizer_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=0.0001, betas=(0.5, 0.999))
#optimizer_EG = torch.optim.Adam([{'params': E.parameters()},{'params': G.parameters(), 'lr': 0.001, 'betas': (0.5, 0.999)}], lr=0.001, betas=(0.5,0.999))


train_loss_nce_list = []
train_loss_gen_list = []

for epoch in range(n_epochs):
    start_time = time.time()

    train_loss_NCE = 0
    train_loss_GEN = 0

    D.train()
    G.train()

    for i, (images, labels) in enumerate(train_data):
        if i > (40000 / batch_size):
            break

        # Sample True
        images = images.cuda()
        images = images.reshape(images.size(0), -1)

        # Sample True Infer
        z = torch.randn(images.shape[0], latent_dim).cuda()
        z_post = langevin_dynamics_generator(z, images)
        G_recon = G(z_post)

        # Sample Fake
        z = torch.randn(images.size(0), latent_dim).cuda()

        # Sample Fake x
        x_gen = G(z)

        loss_r = (torch.log(D(G_recon, z_post) + 1e-6) - torch.log(1 - D(G_recon, z_post) + 1e-6)).sum()
        loss_gen = 1.0 / (2.0 * 0.3 * 0.3) * ((G_recon - images.detach()) ** 2).sum() - loss_r
        optimizer_G.zero_grad()
        loss_gen.backward()
        optimizer_G.step()


        t_acc = D(images, z_post.detach())
        f_acc = 1 - D(x_gen.detach(), z)

        loss_nce = -(torch.log(t_acc + 1e-6) + torch.log(f_acc + 1e-6)).sum()
    #    print(loss_nce)
    #    print("True", t_acc.mean())
    #    print("Fake", f_acc.mean())
        optimizer_D.zero_grad()
        loss_nce.backward()
        optimizer_D.step()

        train_loss_NCE += loss_nce.item()
        train_loss_GEN += loss_gen.item()


    end_time = time.time()
    print('Epoch [{}/{}], GEN_Loss_Train: {:.4f}, NCE_Loss_Train: {:.4f},Time: {:.4f}'.format(epoch + 1, n_epochs, train_loss_GEN / (i+1), train_loss_NCE / (i+1), end_time - start_time))

    train_loss_nce_list.append(train_loss_NCE)
    train_loss_gen_list.append(train_loss_GEN)

    plot_loss_nce(train_loss_nce_list)
    plot_recon(train_loss_gen_list)

    # Test Recon
    images = images.cuda().view(-1, channel*img_dim*img_dim)
    z = torch.randn(images.shape[0], latent_dim).cuda()
    z_post = langevin_dynamics_generator(z, images, False)[:n_show]
    x = images[:n_show]
    for i in range(steps):
        z_post = Variable(z_post, requires_grad=True)
        x = Variable(x, requires_grad=True)
        noise1 = Variable(torch.randn_like(z_post).cuda())
        noise2 = Variable(torch.randn_like(x).cuda())

        r = (torch.log(D(x, z_post) + 1e-6) - torch.log(1 - D(x, z_post) + 1e-6)).sum()
        print("RECON", D(x,z_post).mean())
        z_loss = (z_post ** 2 / 2).sum()
        gen_loss = (1.0 / (2.0 * 0.3 * 0.3) * ((x - G(z_post)) ** 2)).sum()
        loss = -r + z_loss + gen_loss
        loss.backward()

        z_post = z_post - 0.5 * step_size * step_size * (z_post.grad) #+ 0.1 * noise1
        x = x - 0.5 * step_size * step_size * (x.grad)# + 0.1 * noise2
       # x = x.clip(0,1)

    images = images[:n_show].reshape(n_show, img_dim, img_dim, channel)
    x = x.reshape(n_show, img_dim, img_dim, channel).clip(-1,1)

    mse = ((images - x) ** 2).mean()
    '''
    fig, ax = plt.subplots(2, n_show, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0)
    plt.rcParams.update({'font.size': 20})
    fig.suptitle('Epoch {}, MSE {}'.format(epoch + 1, mse))
    fig.text(0.04, 0.5, 'x', ha='left')
    fig.text(0.04, 0.25, 'G(E(x))', ha='left')
    for i in range(n_show):
 #       ax[0, i].imshow(((images[i]*0.5)+0.5).cpu(), cmap='gray')
        ax[0, i].imshow(images[i].cpu())
        ax[0, i].axis('off')
  #      ax[1, i].imshow(((x*0.5)+0.5)[i].cpu().detach().numpy(), cmap='gray')
        ax[1, i].imshow(x[i].cpu().detach().numpy())
        ax[1, i].axis('off')
    plt.savefig(folder + "/recon_correction_%05d.png" % (epoch))
    plt.close()
    '''
    save_image((torch.cat((images[0:8].reshape(-1, channel, img_dim, img_dim), x[0:8].reshape(-1, channel, img_dim, img_dim)), 0)),folder + "/recon_correction_%05d.png" % (epoch))
 #   save_image((torch.cat(((images*0.5+0.5)[0:8].reshape(-1, channel, img_dim, img_dim), (x*0.5+0.5)[0:8].reshape(-1, channel, img_dim, img_dim)), 0)),folder + "/recon_correction_%05d.png" % (epoch))

    # Test Gen
    z = torch.randn(64, latent_dim).cuda()
    x = G(z)

    for i in range(steps):
        z = Variable(z, requires_grad=True)
        x = G(z)
        x = Variable(x, requires_grad=True)
        noise1 = Variable(torch.randn(z.shape[0], 30).cuda())
        noise2 = Variable(torch.randn_like(x).cuda())

        r = (torch.log(D(x, z) + 1e-6) - torch.log(1 - D(x, z) + 1e-6)).sum()
        print("Gen", D(x,z).mean())
        z_loss = (z ** 2 / 2).sum()

        gen_loss = (1.0 / (2.0 * 0.3 * 0.3) * ((x - G(z)) ** 2)).sum()

        loss = -r + z_loss + gen_loss
        loss.backward()

        z = z - 0.5 * step_size * step_size * (z.grad) #+ 0.1 * noise1
        x = x - 0.5 * step_size * step_size * (x.grad)# + 0.1 * noise2
       # x = x.clip(0,1)

    x = x.reshape(64, channel, img_dim, img_dim).clip(0,1)
  #  save_image(((x*0.5)+0.5), folder + "/generation_correction_%05d.png" % (epoch))
    save_image(x, folder + "/generation_correction_%05d.png" % (epoch))

    # Test Recon Pretrain
    z = torch.randn(images.shape[0], latent_dim).cuda()
    z_post = langevin_dynamics_generator(z, images.reshape(-1, img_dim*img_dim*channel), False)[:n_show]
    recon = G(z_post).reshape(n_show, img_dim, img_dim, channel)

    mse = ((images - recon) ** 2).mean()
    save_image((torch.cat((images[0:8].reshape(-1, channel, img_dim, img_dim), recon[0:8].reshape(-1, channel, img_dim, img_dim)), 0)),folder + "/recon_%05d.png" % (epoch))
 #   save_image((torch.cat(((images*0.5+0.5)[0:8].reshape(-1, channel, img_dim, img_dim), (recon*0.5+0.5)[0:8].reshape(-1, channel, img_dim, img_dim)), 0)), folder + "/recon_%05d.png" % (epoch))
    # Test Gen Pretrain
    z = torch.randn(64, latent_dim).cuda()
    gen = G(z).reshape(64, channel, img_dim, img_dim).cpu()

  #  save_image(((gen*0.5)+0.5), folder + "/generation_%05d.png" % (epoch))
    save_image(gen, folder + "/generation_%05d.png" % (epoch))





'''



# Test Recon2
    z = torch.randn(images.shape[0], latent_dim).cuda()
    z_post = langevin_dynamics_generator(z, images.reshape(-1, img_dim*img_dim*channel), False)[:n_show]
    x = images[:n_show].view(-1, img_dim*img_dim*channel)
    for i in range(steps):
        z_post = Variable(z_post, requires_grad=True)
        x = Variable(x, requires_grad=True)
        noise1 = Variable(torch.randn_like(z_post).cuda())
        noise2 = Variable(torch.randn_like(x).cuda())

        r = (torch.log(D(x, z_post) + 1e-6) - torch.log(1 - D(x, z_post) + 1e-6)).sum()
        z_loss = (z_post ** 2 / 2).sum()
        gen_loss = (1.0 / (2.0 * 0.3 * 0.3) * ((x - G(z_post)) ** 2)).sum()

        loss = -r + z_loss + gen_loss
        loss.backward()

        z_post = z_post - 0.5 * step_size * step_size * (z_post.grad) #+ 0.1 * noise1


    images = images[:n_show].reshape(n_show, img_dim, img_dim, channel)
    x = G(z_post)
    x = x.reshape(n_show, img_dim, img_dim, channel)

    mse = ((images - x) ** 2).mean()
   # save_image((torch.cat((images[0:8].reshape(-1, channel, img_dim, img_dim), x[0:8].reshape(-1, channel, img_dim, img_dim)), 0)),folder + "/recon_correction2_%05d.png" % (epoch))
    save_image((torch.cat(((images*0.5+0.5)[0:8].reshape(-1, channel, img_dim, img_dim), (x*0.5+0.5)[0:8].reshape(-1, channel, img_dim, img_dim)), 0)),folder + "/recon_correction2_%05d.png" % (epoch))

    # Test Gen2
    z = torch.randn(64, latent_dim).cuda()
    x = G(z)

    for i in range(steps):
        z = Variable(z, requires_grad=True)
        x = Variable(x, requires_grad=True)
        noise1 = Variable(torch.randn(z.shape[0], 30).cuda())
        noise2 = Variable(torch.randn_like(x).cuda())
        x = G(z)

        r = (torch.log(D(x, z) + 1e-6) - torch.log(1 - D(x, z) + 1e-6)).sum()
        z_loss = (z ** 2 / 2).sum()
        gen_loss = (1.0 / (2.0 * 0.3 * 0.3) * ((x - G(z)) ** 2)).sum()

        loss = -r + z_loss + gen_loss
        loss.backward()

        z = z - 0.5 * step_size * step_size * (z.grad) #+ 0.1 * noise1

    x = G(z)
    x = x.reshape(64, channel, img_dim, img_dim)
    save_image(((x*0.5)+0.5), folder + "/generation_correction2_%05d.png" % (epoch))
  #  save_image(x, folder + "/generation_correction2_%05d.png" % (epoch))
  
s1 = []
for _ in range(int(10000 / 100)):
    z = torch.randn(100, latent_dim).cuda()
    x = G(z).reshape(100, channel, img_dim, img_dim)
    for i in range(steps):
        z = Variable(z, requires_grad=True)
        x = G(z)
        x = Variable(x, requires_grad=True)

        r = (torch.log(D(x, z) + 1e-6) - torch.log(1 - D(x, z) + 1e-6)).sum()
        print("Gen", D(x,z).mean())
        z_loss = (z ** 2 / 2).sum()

        gen_loss = (1.0 / (2.0 * 0.3 * 0.3) * ((x - G(z)) ** 2)).sum()

        loss = -r + z_loss + gen_loss
        loss.backward()

        z = z - 0.5 * step_size * step_size * (z.grad) #+ 0.1 * noise1
        x = x - 0.5 * step_size * step_size * (x.grad)# + 0.1 * noise2
       # x = x.clip(0,1)

    x = x.reshape(100, channel, img_dim, img_dim).clip(0,1)
 #   s1.append((gen*0.5+0.5))
    s1.append(x)
  #  s2.append((gen + 1)/2)
s1 = torch.cat(s1)
fid = fid_score.compute_fid(x_train=None, x_samples=s1, path='/Tian-ds/hli136/project/nce-master/fid_celeba_real/fid_stats_celeba64_train.npz')
print('Fid', fid)

'''