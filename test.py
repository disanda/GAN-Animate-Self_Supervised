import module
import torch
import torchvision

shape = [32, 32, 1]
n_G_upsamplings = 3

# others
#use_gpu = torch.cuda.is_available()
#device = torch.device("cuda" if use_gpu else "cpu")
#G = module.ConvGenerator(128, shape[-1], n_upsamplings=n_G_upsamplings).to(device)

G = module.ConvGenerator(128, shape[-1], n_upsamplings=n_G_upsamplings)
model_dir = './output/pose_gan/checkpoints/Epoch100_batch32.ckpt'
ckpt=torch.load(model_dir,map_location=torch.device('cpu'))#dict
#print(ckpt)
G.load_state_dict(ckpt['G'])
#D_optimizer.load_state_dict(ckpt['D_optimizer'])
#G_optimizer.load_state_dict(ckpt['G_optimizer'])


@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)

for i in range(10):
	z = torch.randn(1, 128, 1, 1).to('cpu')  # a fixed noise for sampling
	x_fake = sample(z)
#print(x_fake.shape)
	torchvision.utils.save_image(x_fake,'./a%d.jpg'%(i))
torchvision.utils.save_image(x_fake,'./%d.jpg'%(2), nrow=10)