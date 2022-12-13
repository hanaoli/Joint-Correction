import os
import sys
import datetime
import math
import shutil
import random
import argparse
import logging
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import copy
# import pygrid
import torchvision.utils as vutils
import torch.distributions as D
from pytorch_gan_metrics import get_inception_score, get_fid

cuda = torch.cuda.is_available()


def gelu(x):
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float())
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



class Deterministic(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, activation=gelu):
        super(Deterministic, self).__init__()

        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_dim)
            self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        out = self.activation(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation(out)

        out = out + x
        return out


class Projection(nn.Module):
    def __init__(self, z_dim, ngf=16, coef=4, use_bn=False, activation=gelu):
        super(Projection, self).__init__()

        self.use_bn = use_bn
        self.activation = activation
        self.ngf = ngf
        self.coef = coef

        self.linear = nn.Linear(z_dim, self.coef * self.ngf * self.ngf)
        self.deconv1 = nn.ConvTranspose2d(self.coef, self.ngf * self.coef, kernel_size=5, stride=1, padding=2, bias=False)

        if self.use_bn:
            self.linear_bn = nn.BatchNorm1d(self.coef * self.ngf * self.ngf)
            self.deconv1_bn = nn.BatchNorm2d(self.ngf * self.coef)

    def forward(self, z):
        out = self.linear(z.view(z.size(0), -1))
        if self.use_bn:
            out = self.linear_bn(out)
        out = self.activation(out)

        out = self.deconv1(out.view(z.size(0), self.coef, self.ngf, self.ngf).contiguous())
        if self.use_bn:
            out = self.deconv1_bn(out)
        out = self.activation(out)
        return out

'''
class G(nn.Module):
    def __init__(self, args, nc=3, ngf=16, coef=4):
        super(G, self).__init__()
        self.args = args
        self.ngf = ngf

        self.projection_layers = nn.ModuleList(
            [Projection(z_dim, ngf=ngf, coef=coef, use_bn=self.args.use_bn) for z_dim in args.z_dims])
        self.deterministic_layers = nn.ModuleList(
            [Deterministic(ngf * coef, ngf * coef, use_bn=self.args.use_bn) for _ in args.z_dims])
        self.deterministic_layers_extra = Deterministic(ngf * coef, ngf * coef, use_bn=self.args.use_bn)

        self.output = nn.ConvTranspose2d(ngf * coef, nc, kernel_size=4, stride=2,
                                         padding=1)  # (64, 16, 16) --> (64, 32, 32)

        self.activation = gelu
    def forward(self, z_top):
        out = z_top
        for i, _ in enumerate(self.args.z_dims):
            out = self.projection_layers[i](out)
            if self.args.use_skip:
                if i > 0:
                    out = self.deterministic_layers[i](out) + out_det
                else:
                    out = self.deterministic_layers[i](out)
                out_det = out
            else:
                out = self.deterministic_layers[i](out)
            # if i == len(self.args.z_dims) - 1:
            #     break
            # out = self.stochastic_layers[i](out, z_lowers[i])
        out = self.deterministic_layers_extra(out)
        out = self.output(out)

        out = F.tanh(out)

        return out
'''

class G(nn.Module):
    def __init__(self, args, nc=3, ngf=16, coef=4):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(args.z_dims[0], 256, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, z):
        x = self.relu(self.conv1(z.view(-1, z.shape[1], 1, 1)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.tanh(self.conv4(x))
        return x


class _netE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # f = nn.LeakyReLU()
        f = nn.GELU()
        self.ebm = nn.Sequential(
            nn.Linear(args.nz, args.ndf),
            f,

            nn.Linear(args.ndf, args.ndf),
            f,

            nn.Linear(args.ndf, args.nez),
        )

    def forward(self, z):
        fx = self.ebm(z.squeeze()).view(-1, self.args.nez)
        return fx


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

class _netEX(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, args.z_dims[0], kernel_size=4, stride=1, padding=0)
        self.relu = nn.LeakyReLU(0.2)

        self.fc1 = nn.Linear(args.z_dims[0] * 2, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)
        self.lrelu = nn.LeakyReLU(0.1)

        f = nn.GELU()
        self.ebm = nn.Sequential(
            nn.Linear(args.nz + args.img_sz * args.img_sz * args.img_ch, args.ndf),
            f,

            nn.Linear(args.ndf, args.ndf),
            f,

            nn.Linear(args.ndf, args.nez),
        )

    def forward(self, z, x):
        x = self.relu(self.conv1(x.view(-1, self.args.img_ch, self.args.img_sz, self.args.img_sz)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        xz = torch.cat([x.view(-1, self.args.z_dims[0]), z.view(-1, self.args.z_dims[0])], dim=1)
        xz = self.lrelu(self.fc1(xz))
        xz = self.lrelu(self.fc2(xz))
        xz = self.fc3(xz)
        return xz

# %%
def infer_z(g, z_var, x, args):
    mse = nn.MSELoss(reduction='sum')
    z = z_var.clone().detach().requires_grad_()

    for i in range(args.z_n_iters):

        x_hat = g(z)
        L = 1.0 / (2.0 * args.z_sigma * args.z_sigma) * mse(x_hat, x)
        z_grads = torch.autograd.grad(L, z, retain_graph=True)
        total_grad = z.data + z_grads[0]
        # total_grad.clamp_(-500, 500)

        z.data = z.data - 0.5 * args.z_step_size * args.z_step_size * (total_grad)

        if args.z_with_noise:
            eps = torch.randn(*z.shape).cuda()
            z.data += args.z_step_size * eps

    z_s_k = z.detach()

    return z_s_k


'''
def infer_z_updatepos(g, z_var, x, E_list, args):
    mse = nn.MSELoss(reduction='sum')
    z = z_var.clone().detach().requires_grad_()
    x0 = g(z).clone().detach().requires_grad_()
    for i in range(args.z_n_iters):
        e_n = 0
        if len(E_list) > 0:
            for index in range(len(E_list)):
                model = E_list[index]
                energy = model(z.squeeze(), x0)
                e_n = e_n + energy

            recon = 1.0 / (2.0 * args.z_sigma * args.z_sigma) * mse(x0, x)
        else:
            recon = 1.0 / (2.0 * args.z_sigma * args.z_sigma) * mse(g(z), x)
        L = -e_n.sum() + recon
        grad = torch.autograd.grad(L, (x0, z), retain_graph=True)
        x_grad = grad[0]
        z_grad = grad[1]

        z.data = z.data - 0.5 * args.z_step_size * args.z_step_size * (z_grad + z.data)
        x0.data = x0.data - 0.5 * args.x_step_size * args.x_step_size * (x_grad)

        if args.z_with_noise:
            z.data += args.z_step_size * torch.randn(*z.shape).cuda()
            #      x0.data += args.x_step_size * torch.randn(*x0.shape).cuda()

    z_s_k = z.detach()

    return z_s_k, x0.clip(-1, 1)
'''


def infer_z_updatepos(g, z_var, x, x_init, E_list, args):
    mse = nn.MSELoss(reduction='sum')
    z = z_var.clone().detach().requires_grad_()
    x_hat = x_init.clone().detach().requires_grad_()
    for i in range(args.z_n_iters):

        x_hat = g(z)
        recon = 1.0 / (2.0 * args.z_sigma * args.z_sigma) * mse(x_hat, x)
        e_n = 0
        for index in range(len(E_list)):
            model = E_list[index]
            energy = model(z.squeeze(),x)
            e_n = e_n + energy
        L = -e_n.sum() + recon
        grad = torch.autograd.grad(L, z, retain_graph=True)
    #    x_grad = grad[0]
        z_grad = grad[0]

        z.data = z.data - 0.5 * args.z_step_size * args.z_step_size * (z_grad + z.data)
   #     x_hat.data = x_hat.data - 0.5 * args.x_step_size * args.x_step_size * (x_grad)
        if args.z_with_noise:
            z.data += args.z_step_size * torch.randn(*z.shape).cuda()


    return z.detach()#, x_hat.clip(-1, 1)

def sample_LD_XZ(z0, x0, args, E_list, g, t=True):
    z0 = z0.clone().detach().requires_grad_()
    x0 = x0.clone().detach().requires_grad_()

    for i in range(args.z_n_iters):
        e_n = 0
        if len(E_list) > 0:
            for index in range(len(E_list)):
                model = E_list[index]
                energy = model(z0.squeeze(), x0)
                e_n = e_n + energy
        recon = (1 / (2 * args.z_sigma * args.z_sigma) * ((-x0 + g(z0))**2)).sum()
        L = -e_n.sum() + recon
        grad = torch.autograd.grad(L, (x0, z0), retain_graph=True)
        x_grad = grad[0]
        z_grad = grad[1]
        z0.data = z0.data - 0.5 * args.z_step_size * args.z_step_size * (z_grad + z0.data)
        x0.data = x0.data - 0.5 * args.x_step_size * args.x_step_size * (x_grad)
     #   x0 = x0.clip(-1,1)
        if t:
            z0.data += args.z_step_size * torch.randn(*z0.shape).cuda()
      #      x0.data += args.x_step_size * torch.randn(*x0.shape).cuda()

    return z0.detach(), x0.clip(-1, 1)
#

########################################################################################################


def valueX(energy, pos_sample, noise_sample, x_hat, x):
    #    logp_x = energy(pos_sample)
    #    logq_x = p0.log_prob(pos_sample.squeeze()).unsqueeze(1)
    #    logp_gen = energy(noise_sample)
    #    logq_gen = p0.log_prob(noise_sample.squeeze()).unsqueeze(1)

    log_ratio_pos = energy(pos_sample, x)
    log_ratio_neg = energy(noise_sample, x_hat)

    ll_data = log_ratio_pos - torch.logsumexp(torch.cat([log_ratio_pos, torch.zeros_like(log_ratio_pos)], dim=1), dim=1,
                                              keepdim=True)
    ll_gen = - torch.logsumexp(torch.cat([log_ratio_neg, torch.zeros_like(log_ratio_neg)], dim=1), dim=1, keepdim=True)

    v = ll_data.mean() + ll_gen.mean()

    return -v
# %%
def parse_args():
    args = argparse.Namespace()

    args.seed = 1
    args.device = 2

    args.n_epochs = 101
    args.new_stage_every = 101

    args.img_sz = 32
    args.img_ch = 3
    args.n_batch = 100
    args.lr = 1e-4

    args.use_bn = False
    args.use_ladder = True
    args.use_skip = True

    args.z_dims = [256]
    args.z_sigma = 0.3
    args.z_step_size = 0.3
    args.z_n_iters = 30
    args.x_step_size = 0.05
    args.z_step_size_prior = 0.3
    args.z_n_iters_prior = 30
    args.z_with_noise = True

    args.nz = 256
    args.e_sn = False

    args.ndf = 200
    args.nez = 1
    args.lr_e = 4e-5

    args.e_gamma = 0.998
    args.g_gamma = 0.998

    return args

class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, G, args):
        self.G = G
        self.args = args

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        return (self.G(torch.randn(1, self.args.nz, 1, 1).cuda()) + 1)/2


def train(args_job, output_dir_job, output_dir, return_dict):
    args = parse_args()

  #  set_gpu(args.device)
    set_seed(args.seed)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    transform_celeba = transforms.Compose([  # transforms.CenterCrop(120),
                                            #  transforms.Resize(args.img_sz),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ])

    ds_train = torch.utils.data.DataLoader(
                torchvision.datasets.SVHN(root='./data/SVHN', split='train', download=True,
                                           transform=transform_celeba),
                batch_size=args.n_batch, shuffle=False)


    g = G(args).cuda()

    optim = torch.optim.Adam(g.parameters(), lr=args.lr, betas=[.5, .999])

    E_list = []

 #   E = _netE(args).cuda()
    E = _netEX(args).cuda()
    optim_c = torch.optim.Adam(E.parameters(), lr=args.lr_e, betas=(0.5, 0.999))

    mse = nn.MSELoss(reduction='sum')
    lr_scheduleG = torch.optim.lr_scheduler.ExponentialLR(optim, args.e_gamma)
    lr_scheduleE = torch.optim.lr_scheduler.ExponentialLR(optim_c, args.g_gamma)
   # E_list.append(copy.deepcopy(E))
    min_fid = 1000
    min_fid_cor_nn = 1000
    min_fid_cor_nn_z = 1000

    Z_new = torch.empty(len(ds_train.dataset), args.nz, 1, 1)
    X_new = torch.empty(len(ds_train.dataset), args.img_ch, args.img_sz, args.img_sz)
    for epoch in range(args.n_epochs + 1):
        for i, (x, ind) in enumerate(ds_train):
            if i > (70000 / args.n_batch):
                break
            batch_size = x.size(0)
            x = x.cuda()
    #        x = F.interpolate(x, size=32, mode='bilinear')
            x = x.clamp(-1, 1)

            z_init = torch.randn(batch_size, args.nz, 1, 1).cuda()
            if len(E_list) == 0:
                z = infer_z(g, z_init, x, args)
                x_hat = g(z)
            else:
           #     z_init = Z_old[i*ds_train.batch_size:(i+1)*ds_train.batch_size].cuda()
            #    x_init = X_old[i*ds_train.batch_size:(i+1)*ds_train.batch_size].cuda()
           #     z, x_hat = infer_z_updatepos(g, z_init, x, g(z_init), E_list, args)
                z = infer_z_updatepos(g, z_init, x, g(z_init), E_list, args)
                x_hat = g(z)

       #     Z_new[i*ds_train.batch_size:(i+1)*ds_train.batch_size] = z.clone().cpu()
      #      X_new[i*ds_train.batch_size:(i+1)*ds_train.batch_size] = x_hat.clone().cpu()

            mse_hat = mse(x_hat, x)
            L = 1 / (2 * args.z_sigma * args.z_sigma) * mse_hat

            optim.zero_grad()
            L.backward()
            optim.step()

            optim_c.zero_grad()


            z_init = torch.randn(batch_size, args.nz, 1, 1).cuda()
            x_init = g(z_init)
            if len(E_list) > 0:
             #   z_prior = Z_old[ind].cuda()
             #   x_prior = X_old[ind].cuda()
               # x_prior = g(z_prior)
                z_init, x_init = sample_LD_XZ(z_init, x_init, args, E_list, g)

         #   if len(E_list) == 0:
         #       x_init = g(z_init)
         #   else:
         #       z_init, x_init = sample_LD_XZ(z_init, g(z_init), args, E_list, g, t=True)
            loss_energy = valueX(E, z, z_init, x_init.detach(), x)
            loss_energy.backward()

            optim_c.step()

            if i % 100 == 0:
                print('[%3d/%3d][%3d/%3d] errG: %10.2f loss_energy: %.4f'
              % (epoch, args.n_epochs, i, len(ds_train),
                 (mse_hat.data / args.n_batch).item(), loss_energy.item()
                 ))
        if epoch % 5 == 0 and epoch > 15:
            import fid_score
            s1 = []
            for _ in range(int(10000 / 100)):
                z = torch.randn(100, args.nz).cuda()
                x = g(z).reshape(100, 3, 32, 32)
                x = (x + 1) / 2
                s1.append(x)
            s1 = torch.cat(s1)
            fid = fid_score.compute_fid(x_train=None, x_samples=s1,
                                        path='/Tian-ds/hli136/project/nce-master/fid_real/fid_stats_svhn_train.npz')
            print('Fid', fid)

            if fid < min_fid:
                min_fid = fid
                os.makedirs(output_dir + '/ckpt', exist_ok=True)
                save_dict = {
                    'epoch': epoch,
                    'fid': min_fid,
                    'netG': g.state_dict(),
                }
                torch.save(save_dict, '{}/Generator.pth'.format(output_dir + '/ckpt'))
            print('Min Fid', min_fid)
            '''
            s1 = []
            for _ in range(int(40000 / 100)):
                z = torch.randn(100, args.nz).cuda()
                if len(E_list) == 0:
                    z = sample_LD_1(z, args.z_step_size_prior, args.z_n_iters_prior, E, g)
                else:
                    z = sample_LD(z, args.z_step_size_prior, args.z_n_iters_prior, E_list, g)
                x = g(z).reshape(100, 3, 32, 32)
                x = (x + 1) / 2
                s1.append(x)
            s1 = torch.cat(s1)
            fid = fid_score.compute_fid(x_train=None, x_samples=s1,
                                        path='/Tian-ds/hli136/project/nce-master/fid_real/fid_stats_svhn_train.npz')
            print('Fid2', fid)
            if fid < min_fid_cor:
                min_fid_cor = fid
            print('Min Fid Cor', min_fid_cor)
                '''

            s1 = []
            s2 = []
            if len(E_list) > 0:
                for _ in tqdm(range(int(10000 / 100))):
                    z = torch.randn(100, args.nz).cuda()
                    x = g(z)
                    z0, x = sample_LD_XZ(z, x, args, E_list, g)
                    x = (x + 1) / 2
                    x2 = (g(z0) + 1) / 2
                    s1.append(x)
                    s2.append(x2)
                s1 = torch.cat(s1)
                s2 = torch.cat(s2)
                fid1 = fid_score.compute_fid(x_train=None, x_samples=s1,
                                             path='/Tian-ds/hli136/project/nce-master/fid_real/fid_stats_svhn_train.npz')
                print('Fid3', fid1)
                if fid1 < min_fid_cor_nn:
                    min_fid_cor_nn = fid1
                print('Min Fid Cor X*', min_fid_cor_nn)

                fid2 = fid_score.compute_fid(x_train=None, x_samples=s2,
                                             path='/Tian-ds/hli136/project/nce-master/fid_real/fid_stats_svhn_train.npz')
                print('Fid3', fid2)
                if fid2 < min_fid_cor_nn_z:
                    min_fid_cor_nn_z = fid2
                print('Min Fid Cor Z*', min_fid_cor_nn_z)


            lr_scheduleE.step()
            lr_scheduleG.step()

            gen_samples = g(z).detach()
            vutils.save_image(gen_samples.data, '%s/epoch_%03d_samples.png' % (output_dir, epoch), normalize=True,
                              nrow=int(np.sqrt(z[0].shape[0])))

            print("LD: -1+1")
            vutils.save_image(x.data, '%s/epoch2_%03d_samples.png' % (output_dir, epoch), normalize=True,
                              nrow=int(np.sqrt(z[0].shape[0])))

        if epoch % args.new_stage_every == 0 and epoch > 0:
            E_list.append(copy.deepcopy(E))
            E = _netEX(args).cuda()
            optim_c = torch.optim.Adam(E.parameters(), lr=args.lr_e, betas=(0.5, 0.999))
            lr_scheduleE = torch.optim.lr_scheduler.ExponentialLR(optim_c, args.e_gamma)

            Z_old = Z_new.clone()
            Z_new = torch.empty(len(ds_train.dataset), args.nz, 1, 1)

            X_old = X_new.clone()
            X_new = torch.empty(len(ds_train.dataset), args.img_ch, args.img_sz, args.img_sz)

        if epoch > 0 and epoch % 25 == 0:
            torch.save(g.state_dict(), '%s/g_epoch_%d.pth' % (output_dir, epoch))
            if len(E_list) == 0:
                torch.save(E.state_dict(), '%s/E_epoch_%d.pth' % (output_dir, epoch))
            else:
                E_dict = {}
                for E_index in range(len(E_list)):
                    E_dict['E_{}'.format(E_index)] = E_list[E_index].state_dict()
                torch.save(E_dict, '%s/E_epoch_%d.pth' % (output_dir, epoch))


##############################################################################

def set_seed(seed=None):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def set_gpu(gpu):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True

        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


##############################################################################

import itertools


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def to_named_dict(ns):
    d = AttrDict()
    for (k, v) in zip(ns.__dict__.keys(), ns.__dict__.values()):
        d[k] = v
    return d


def merge_dicts(a, b, c):
    d = {}
    d.update(a)
    d.update(b)
    d.update(c)
    return d


def main():
    #
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = get_output_dir(exp_id)

    # run
    copy_source(__file__, output_dir)
    # opt = create_opts()[0]
    opt = {'job_id': int(0), 'status': 'open', 'device': 0}
    train(opt, output_dir, output_dir, {})


if __name__ == '__main__':
    main()
