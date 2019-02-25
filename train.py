%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser(description='Flower Classifcation trainer')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or not')
    parser.add_argument('--arch', type=str, default='densenet', help='architecture [available: densenet, vgg]', required=True)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=100, help='hidden units for fc layer')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='path of your saved model')
    args = parser.parse_args()

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

def flower_data(args):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
    image_datasets['valid'] = datasets.ImageFolder(data_dir + '/test', transform = test_transforms)
    image_datasets['test'] = datasets.ImageFolder(data_dir + '/valid', transform = valid_transforms)
    
    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(train_datasets, batch_size=32)
    dataloaders['valid'] = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    dataloaders['test'] = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    
    return image_datasets, dataloaders

def build_network(args):
    
    image_datasets, dataloaders = flower_data(args)
    
    if args.arch == 'vgg':
        model = models.vgg16(pretrained=True)
        
    elif args.arch == 'densenet':
        model = models.densenet121(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    input_size = model.classifier[0].in_features
    output_size = 102
    hidden_size = [3136,784,102]
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, args.hidden_size[0])),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(args.hidden_size[0], args.hidden_size[1])),
        ('relu2', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2)),
        ('output', nn.Linear(args.hidden_size[1], output_size)),
        ('softmax', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier= classifier
    
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Using GPU: "+ str(use_gpu))
        else:
            print("Using CPU since GPU is not available/configured")
            
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.lr=0.001)
    
    epochs = 5
    running_loss = 0
    print_every = 50
    steps = 0
    
    for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(train_dataloaders):
        steps += 1
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0
            
     model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': classifier,
        'class_to_idx': model.class_to_idx,
    }
    
    torch.save(checkpoint, args.saved_model)
    
    build_network(args)
    
    
if __name__ == "__main__":
    main()
