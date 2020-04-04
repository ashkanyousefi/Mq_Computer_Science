import json
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.autograd import Variable
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', action='store', help='data directory where images are stored')
    parser.add_argument('-a', '--arch', dest='arch', default='vgg16', choices=['vgg16', 'vgg19'],help='model architexture to use')
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', default='0.001',help='learning rate for gradient descent')
    parser.add_argument('-u', '--hidden_units', dest='hidden_units', default='512',
    help='number of hidden units the classifier will be using')
    parser.add_argument('-e', '--epochs', dest='epochs', default='5',
                        help='number of epochs the training model will loop through')
    parser.add_argument('-g', '--gpu', action="store_true", default=True,
                        help='specify if processing on gpu is preferred')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    dataloaders_train, dataloaders_valid, dataloaders_test,class_to_idx  = image_transformation(data_dir,train_dir,valid_dir,test_dir)
    model=create_model(arch, hidden_units )
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    criterion = nn.NLLLoss()
    
    train(model, epochs, learning_rate, criterion, optimizer, dataloaders_train, dataloaders_valid, gpu)
     
    accuracy = accuracy_calculation()

    model.class_to_idx = class_to_idx

    save_checkpoint(model, optimizer, arch)
#processed_image = process_image(image_path)

def image_transformation(data_dir,train_dir,valid_dir,test_dir):
    data_transforms_train = transforms.Compose([transforms.Resize(255), transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30), 
        transforms.RandomCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transforms_test = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets_train = datasets.ImageFolder(train_dir, transform = data_transforms_train)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform = data_transforms_test)
    image_datasets_test = datasets.ImageFolder(test_dir, transform = data_transforms_test)

    dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size = 64, shuffle = True)
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size = 64)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size = 64)

    class_to_idx = image_datasets_train.class_to_idx

    return dataloaders_train, dataloaders_valid, dataloaders_test, class_to_idx

def create_model(model, hidden_units):

    if (model == "vgg16"):
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),  
        ('relu', nn.ReLU()),  
        ('fc2', nn.Linear(hidden_units, 102)), 
        ('output', nn.LogSoftmax(dim=1))  
    ]))

    model.classifier = classifier
    return model

    # return "Model Created Successfully"

def train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader, gpu):
    epochs = epochs
    model.train()

    cuda = torch.cuda.is_available()
    if cuda and gpu:
        model.cuda() 
    else:
        model.cpu()

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(dataloaders_train):

            if cuda and gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)  

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            validation_loss, accuracy = validate(model, criterion, validation_loader)

            print("Epoch: {}/{} ".format(epoch+1, epochs),
                  "Training Loss: {:.3f} ".format(running_loss),
                  "Validation Loss: {:.3f} ".format(validation_loss),
                  "Validation Accuracy: {:.3f}".format(accuracy))

def validate(model, criterion, data_loader):
    model.eval()
    accuracy = 0
    test_loss = 0
    cuda = torch.cuda.is_available()

    for inputs, labels in iter(data_loader):

        if cuda and gpu:
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True)
        else:
            inputs=Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)

def accuracy_calculation():
    criterion = nn.NLLLoss()
    cuda = torch.cuda.is_available()
    
    if cuda and gpu:
        model.cuda()
    else:
        model.cpu()
    
    for images, labels in dataloaders_test:
        if cuda and gpu:
            images = Variable(images.float().cuda())
            labels = Variable(labels.long().cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

    loss_test = 0
    output = model(images)
    loss_t = criterion(output, labels)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)
    loss_test += loss_t
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
    
    return accuracy

 
def save_checkpoint(model, optimizer, arch):
    model.cpu()
    torch.save({'arch': arch, 
            'model':model,
            'state_dict': model.state_dict(), 
            'classifier' : model.classifier,
            'optimizer': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx},
            'saved_file.pth')



# def load_model(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)

# #     model = models.vgg19(pretrained=True)

#     for param in model.parameters():
#         param.requires_grad = False

#     model.class_to_idx = checkpoint['class_to_idx']

# #     classifier = nn.Sequential(OrderedDict([
# #                           ('fc1', nn.Linear(25088, 4096)),
# #                           ('relu', nn.ReLU()),
# #                           ('fc2', nn.Linear(4096, 102)),
# #                           ('output', nn.LogSoftmax(dim=1))
# #                           ]))

# #     model.classifier = classifier

#     model.load_state_dict(checkpoint['state_dict'])

#     return model


# # In[17]:


# model = load_model('ashlaki_R2.pth')
# model

# def imshow(image, ax=None, title=None):
#     """Imshow for Tensor."""
#     if ax is None:
#         fig, ax = plt.subplots()

#     image = image.transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
#     image = np.clip(image, 0, 1)
#     ax.imshow(image)

#     return ax
