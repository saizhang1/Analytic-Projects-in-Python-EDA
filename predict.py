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

use_gpu = torch.cuda.is_available

def main():

    parser = argparse.ArgumentParser(description='Flower Classification Predictor')
    parser.add_argument('--gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('--image_path', type=str, help='path of image')
    parser.add_argument('--hidden_units', type=int, default=100, help='hidden units for fc layer')
    parser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='path of your saved model')
    parser.add_argument('--mapper_json' , type=str, default='cat_to_name.json', help='path of your mapper from category to name')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')

    args = parser.parse_args()


    import json
    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)
        
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    pil_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    pil_image = Image.open(image)
    pil_image = pil_transforms(pil_image).float()  
    
    return pil_image

def load_checkpoint (file='checkpoint.pth'):
    checkpoint_provided = torch.load(args.saved_model)
    if checkpoint_provided['arch'] == 'vgg':
        model = models.vgg16()        
    elif checkpoint_provided['arch'] == 'densenet':
        model = models.densenet121()
        
        
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
        
    model.classifier = classifier
    model.load_state_dict(checkpoint_provided['state_dict'])
    
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Using GPU")
        else:
            print("Using CPU since GPU is not available/configured")

    class_to_idx = checkpoint_provided['class_to_idx']
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class
        

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    pil_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    pil_image = Image.open(image)
    pil_image = pil_transforms(pil_image).float()  
    
    return pil_image
     
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax  
        
        
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.cpu()

    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(image)
        top_probability, top_labels = torch.topk(output, topk)
        
        top_probability = top_probability.exp()
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_class = list()
    
    for label in top_labels.numpy()[0]:
        top_class.append(class_to_idx_inv[label])
        
    return top_probability.numpy()[0], top_class
        
    model, class_to_idx, idx_to_class = load_checkpoint(args)
    top_probability, top_class = predict(args, args.image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=args.topk)
                                              
    print('Predicted Classes: ', top_class)
    print ('Class Names: ')
    [print(cat_to_name[x]) for x in top_class]
    print('Predicted Probability: ', top_probability)
     
if __name__ == "__main__":
    main()
 

    