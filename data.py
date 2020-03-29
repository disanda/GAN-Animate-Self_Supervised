import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class DatasetFromFolder(Dataset):
    def __init__(self):
        super().__init__(path='./img_align_celeba',size=32)
        self.path = path#指定自己的路径
        #self.image_filenames = [x for x in os.listdir(self.path) if x.endswith('jpg') or x.endswith('png')] # x.startswith() 
        imgs_path = os.listdir('data/img_align_celeba')
        self.image = list(filter(lambda x:x.endswith('jpg') or x.endswith('png') ,imgs_path))
    def __getitem__(self, index):
        a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('L')
        a = a.resize((size, size), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        return a
    def __len__(self):
        return len(self.image_filenames)

def make_dataset(dataset_name, batch_size, drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False,img_paths=''):
    if dataset == 'mnist' or 'fashion_mnist':
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        if dataset == 'mnist':
            dataset = datasets.MNIST('data/MNIST', transform=transform, download=True)
        else:
            dataset = datasets.FashionMNIST('data/FashionMNIST', transform=transform, download=True)
        img_shape = [32, 32, 1]
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = datasets.CIFAR10('data/CIFAR10', transform=transform, download=True)
        img_shape = [32, 32, 3]
    elif dataset == 'pose'：
            transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = DatasetFromFolder(path='')
        img_shape = [32, 32, 1]
    elif dataset == 'celeba_64'
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]# [image,height,width]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.Resize(size=(64, 64)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            #transforms.ToPILImage()
            ])
        dataset = torchlib.DatasetFromFolder(path='',size=64)
        img_shape = (64, 64, 3)
    else:
        raise NotImplementedError
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)
    return data_loader, img_shape

# ==============================================================================
# =                                   debug                                    =
# ==============================================================================

# pose = DatasetFromFolder()

# train_loader = torch.utils.data.DataLoader(
#      dataset=pose,
#      batch_size=25,#一个batch25张图片,epoch=allData_size/batch_size
#      shuffle=False,
#      #num_workers=0,若是win需要这一行
#      pin_memory=True,#用Nvidia GPU时生效
#      drop_last=True
#  )

# for i, x in enumerate(train_loader):
#      print(i)
#      print(x.shape)#[n,c,w,h]
# torchvision.utils.save_image(x, './pose-img/%d.jpg'%(i), nrow=5)
