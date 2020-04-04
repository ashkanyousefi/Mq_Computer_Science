
#Imports
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets,models,transforms
from PIL import Image
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.autograd import Variable
import argparse
from utils import load_checkpoint, load_cat_names
import json
 
def main():  
    args = parse_args()
    checkpoint=args.checkpoint
    image_path=args.image_path
    category_names=args.category_names
    gpu=args.gpu
    top_k=args.top_k

    model = load_checkpoint(checkpoint)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    probs, classes = predict(image_path, model, top_k, gpu)

    print('\n======')
    print('The filepath of the selected image is: ' + image_path, '\n')    
    print('The top K CLASSES for the selected image are: \n', classes, '\n')
    print('The top K PROBABILITIES for the selected image are: \n ', probs, '\n')   
    print('The top K CATEGORY NAMES for the selected image are: \n', [cat_to_name[x].title() for x in classes])
    print('======\n')

    if __name__ == "__main__":
        main()


def parse_args():
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument('--checkpoint', action='store', default='ashlaki.pth')
    parser.add_argument('-t', '--top_k', dest='top_k', default='1',
                       help='number of top probabilities - default: 1')
    parser.add_argument('-f', '--image_path', dest='image_path', default=None,
                       help='path to image file for processing')
    parser.add_argument('-c', '--category_names', dest='category_names', default='cat_to_name.json',
                       help='json file with categories/classes to real name mapping')
    parser.add_argument('-g', '--gpu', action='store_true', default=True,
                       help='specify if processing on gpu is preferred')
    return parser.parse_args()

def process_image(image_path):

    img=Image.open(image_path)
    img=img.resize((256,256))
    value=0.5*(256-224)
    img=img.crop((value,value,256-value,256-value))
    img=np.array(img)/255
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    img=(img-mean)/std
    
    return img.transpose(2,0,1)

def predict(image_path, model, topk, gpu):

    img = process_image(image_path)
    img = torch.from_numpy(np.array([img])).float()

    cuda = torch.cuda.is_available()
    if cuda and gpu:
        model.cuda() 
        img = img.cuda()
    else:
        model.cpu()
        
    output = model.forward(img)
    log_out = torch.exp(output)
    top_p,top_class = torch.topk(log_out, topk)
    idx_to_class={}
    for key,value in model.class_to_idx.items():
        idx_to_class[value]=key
    
    np_top_class=np.array(top_class[0])    
    top_class_name=[]
    for labels in np_top_class:
        top_class_name.append(int(idx_to_class[labels]))
    
    top_flowers = [cat_to_name[str(lab)] for lab in top_class_name]
    
    # for x in top_class.tolist()[0]:
    #     if x>0:
    #         top_class_name.append(cat_to_name[str(x)])
    return top_p, top_flowers

def load_model(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    model=checkpoint['model']

    for param in model.parameters():
        param.requires_grad = False

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    optimizer = checkpoint['optimizer']
    
    return model
