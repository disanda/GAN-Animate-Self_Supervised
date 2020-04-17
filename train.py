
import functools
import numpy as np
import pylib as py
import tensorboardX
import torch
import loss
import gradient_penalty as gp
import tqdm

import data
import module
import torchvision

import os

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

import argparse
parser = argparse.ArgumentParser(description='the training args')
parser.add_argument('--dataset_name',default='pose')#choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'pose']
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0002,help='learning_rate')
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--n_d', type=int, default=1)# d updates per g update
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', '1-gp', '0-gp', 'lp'])
parser.add_argument('--gradient_penalty_sample_mode', default='line', choices=['line', 'real', 'fake', 'dragan'])
parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
parser.add_argument('--experiment_name', default='none')
parser.add_argument('--gradient_penalty_d_norm', default='layer_norm', choices=['instance_norm', 'layer_norm'])
parser.add_argument('--img_size',type=int,default=32)
args = parser.parse_args()

# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s_%_%s' % (args.dataset_name, args.adversarial_loss_mode,args.batch_size,args.epochs)
    if args.gradient_penalty_mode != 'none':
        experiment_name += '_%s_%s' % (args.gradient_penalty_mode, args.gradient_penalty_sample_mode)
output_dir = os.path.join('output', args.experiment_name)

if not os.path.exists('output'):
    os.mkdir('output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
#os.mkdirs(output_dir)

# save settings
import yaml
with open(os.path.join(output_dir, 'settings.yml'), "w", encoding="utf-8") as f:
    yaml.dump(args, f)

# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# setup dataset
if args.dataset_name in ['cifar10', 'fashion_mnist', 'mnist']:  # 32x32
    data_loader, shape = data.make_dataset(args.dataset_name, args.batch_size,args.img_size,pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 3

elif args.dataset_name == 'celeba':  # 64x64
    data_loader, shape = data.make_dataset(args.dataset_name, args.batch_size,args.img_size,pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset_name.find('pose') != -1:  # 32x32
    #img_paths = os.listdir('data/pose')
    #img_payhs = list(filter(lambda x:x.endswith('png'),img_paths))
    data_loader, shape = data.make_dataset(args.dataset_name,args.batch_size,arg.img_size,pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4  # 3 for 32x32 and 4 for 64x64

# ==============================================================================
# =                                   model                                    =
# ==============================================================================

# setup the normalization function for discriminator
if args.gradient_penalty_mode == 'none':
    d_norm = 'batch_norm'
else:  # cannot use batch normalization with gradient penalty
    d_norm = args.gradient_penalty_d_norm

# networks
G = module.ConvGenerator(args.z_dim, shape[-1], n_upsamplings=n_G_upsamplings).to(device)
D = module.ConvDiscriminator(shape[-1], n_downsamplings=n_D_downsamplings, norm=d_norm).to(device)
print(G)
print(D)

# adversarial_loss_functions
d_loss_fn, g_loss_fn = loss.get_adversarial_losses_fn(args.adversarial_loss_mode)

# optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

def train_G():
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z)

    x_fake_d_logit = D(x_fake)
    G_loss = g_loss_fn(x_fake_d_logit)

    G.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    return {'g_loss': G_loss}


def train_D(x_real):
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z).detach()

    x_real_d_logit = D(x_real)
    x_fake_d_logit = D(x_fake)

    x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
    gp_value = gp.gradient_penalty(functools.partial(D), x_real, x_fake, gp_mode=args.gradient_penalty_mode, sample_mode=args.gradient_penalty_sample_mode)

    D_loss = (x_real_d_loss + x_fake_d_loss) + gp_value * args.gradient_penalty_weight

    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp_value}

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# load checkpoint if exists
ckpt_dir = os.path.join(output_dir, 'checkpoints')
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
try:
    ckpt_path = os.path.join(ckpt_dir, 'xxx.ckpt')
    ckpt=torch.load(ckpt_path)
    ep, it_d, it_g = ckpt['ep'], ckpt['it_d'], ckpt['it_g']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    D_optimizer.load_state_dict(ckpt['D_optimizer'])
    G_optimizer.load_state_dict(ckpt['G_optimizer'])
except:
    ep, it_d, it_g = 0, 0, 0

# sample
sample_dir = os.path.join(output_dir, 'samples_training')
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

# main loop
writer = tensorboardX.SummaryWriter(os.path.join(output_dir, 'summaries'))
z = torch.randn(100, args.z_dim, 1, 1).to(device)  # a fixed noise for sampling

@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)

for ep_ in tqdm.trange(args.epochs):#epoch:n*batch
    ep = ep+1
    for x_real in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):#batch_size
        if dataset_name == 'cifar10':#数据有标签
            images,labels = x_real
            images = images.to(device)
        else:
            images = x_real
            images = images.to(device)

        D_loss_dict = train_D(images)
        it_d += 1
        for k, v in D_loss_dict.items():
            writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=it_d)

        if it_d % args.n_d == 0:
            G_loss_dict = train_G()
            it_g += 1
            for k, v in G_loss_dict.items():
                writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=it_g)

        # sample
        if it_g % 100 == 0:
            x_fake = sample(z)
            #x_fake = np.transpose(x_fake.data.cpu().numpy(), (0, 2, 3, 1))#(n,w,h,c)
            torchvision.utils.save_image(x_fake,sample_dir+'/%d.jpg'%(it_g), nrow=10)
    # save checkpoint
    if ep*10 == 0:
        torch.save({'ep': ep, 'it_d': it_d, 'it_g': it_g,
                              'D': D.state_dict(),
                              'G': G.state_dict(),
                              'D_optimizer': D_optimizer.state_dict(),
                              'G_optimizer': G_optimizer.state_dict()},
                              os.path.join(ckpt_dir, '/Epoch_(%d).ckpt' % ep)
                              )
