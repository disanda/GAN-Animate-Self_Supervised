# coding=utf-8
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class DatasetFromFolder(Dataset):
    def __init__(self,path='',size=32):
        super().__init__()
        self.path = path
        self.size = size
        self.image_filenames = [x for x in os.listdir(self.path) if x.endswith('jpg') or x.endswith('png')] # x.startswith() 
        #imgs_path = os.listdir(path)
        #self.image_filenames = list(filter(lambda x:x.endswith('jpg') or x.endswith('png') ,imgs_path))
    def __getitem__(self, index):
        a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('L')
        a = a.resize((self.size, self.size), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        return a
    def __len__(self):
        return len(self.image_filenames)

def make_dataset(dataset_name, batch_size, drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False,img_paths=''):
    if dataset_name == 'mnist' or 'fashion_mnist':
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        if dataset_name == 'mnist':
            dataset = datasets.MNIST('data/MNIST', transform=transform, download=True)
        else:
            dataset = datasets.FashionMNIST('data/FashionMNIST', transform=transform, download=True)
        img_shape = [32, 32, 1]
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = datasets.CIFAR10('data/CIFAR10', transform=transform, download=True)
        img_shape = [32, 32, 3]
    elif dataset_name == 'pose':
            transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
            dataset = DatasetFromFolder(path='/_yucheng/dataSet/pose_set_1')
            img_shape = [32, 32, 1]
    elif dataset_name == 'celeba_64':
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

# pose = DatasetFromFolder('/Users/apple/Desktop/AI_code/dataSet/pose_set_1000')

# train_loader = DataLoader(
#      dataset=pose,
#      batch_size=25,#一个batch25张图片,epoch=allData_size/batch_size
#      shuffle=False,
#      #num_workers=0,若是win需要这一行
#      pin_memory=True,#用Nvidia GPU时生效
#      drop_last=True
#  )

# import tqdm
# for x_real in tqdm.tqdm(train_loader, desc='Inner Epoch Loop'):
#     print(type(x_real))

# for i, x in enumerate(train_loader):
#      print(i)
#      print(x.shape)#[n,c,w,h]
# torchvision.utils.save_image(x, './pose-img/%d.jpg'%(i), nrow=5)
