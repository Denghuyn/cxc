import cv2
import torch
import torchvision
import torchvision.datasets as datasets
import albumentations as A
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os

data_dir="/media/mountHDD3/data_storage/z2h/chestX_ray/data/chest_xray/chest_xray"

def loadTrainData(data_dir, args, resize=(224,224)):
  data_transforms={
      'train': transforms.Compose([
          transforms.RandomResizedCrop(max(resize)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485,0.486,0.406],[0.229,0.224,0.225])
      ]),
      'val': transforms.Compose([
          transforms.RandomResizedCrop(max(resize)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485,0.486,0.406],[0.229,0.224,0.225])
      ]),
      'test': transforms.Compose([
          transforms.RandomResizedCrop(max(resize)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485,0.486,0.406],[0.229,0.224,0.225])
      ])
      
  }
  dataset={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
          for x in ['train','test','val']}
  dataset_loader={x:torch.utils.data.DataLoader(dataset[x],
                                                batch_size=args.bs,
                                               shuffle=True)
                 for x in ['train','test','val']}
  dataset_size={x:len(dataset[x]) for x in ['train','test','val']}
  dataset_classes=dataset['train'].classes
  inputs, classes = next(iter(dataset_loader['train']))
  return args, dataset_loader['train'],dataset_loader['val'],dataset_loader['test'], inputs, classes