#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,models,transforms
from PIL import Image
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict

from torch.optim import lr_scheduler
from torch.autograd import Variable


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms_train = transforms.Compose([transforms.Resize(255),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomRotation(30),
                                        transforms.RandomCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ])

data_transforms_test = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                             ])


                                                                                        
# TODO: Load the datasets with ImageFolder
image_datasets_train = datasets.ImageFolder( train_dir, transform=data_transforms_train)
image_datasets_valid = datasets.ImageFolder( valid_dir, transform=data_transforms_test)
image_datasets_test = datasets.ImageFolder( test_dir, transform=data_transforms_test)


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders_train = torch.utils.data.DataLoader(image_datasets_train,batch_size=64,shuffle=True)
dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid,batch_size=64)
dataloaders_test = torch.utils.data.DataLoader(image_datasets_test,batch_size=64)

class_to_idx = image_datasets_train.class_to_idx


# In[4]:


print(class_to_idx)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[5]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[6]:


# model_selection=input('Please select VGG16 or VGG19:')
# print(model_selection)

# comp_selection=input('Please select CPU or GPU:')
# print(comp_selection)


# In[7]:


# # TODO: Build and train your network

# device = torch.device("cuda" if comp_selection.lower()=='gpu' else "cpu")

# if (model_selection.lower()=="vgg16"):
model=models.vgg16(pretrained=True)
# else:
#     model=models.vgg19(pretrained=True)


# In[8]:


for param in model.parameters():
    param.requires_grad=False

    
classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(25088, 4096)), # First layer
                      ('relu', nn.ReLU()), # Apply activation function
                      ('fc2', nn.Linear(4096, 102)), # Output layer
                      ('output', nn.LogSoftmax(dim=1)) # Apply loss function
                      ]))

    
model.classifier=classifier


# In[ ]:


# import sys 
  
# print("This is the name of the program:", sys.argv) 
  
# print("Argument List:", str(sys.argv)) 


# In[10]:


def validate(model, criterion, data_loader):
    model.eval() # Puts model into validation mode
    accuracy = 0
    test_loss = 0
    
    for inputs, labels in iter(data_loader):
        if torch.cuda.is_available():
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True) 
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output).data 
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)


# In[11]:


def train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader,comp_mode):
    
    model.train() # Puts model into training mode
    use_gpu = False
    
    # Check to see whether GPU is available
    if comp_mode.lower()=="gpu":
        use_gpu = True
        model.cuda()
    else:
        model.cpu()
        
    
     # Iterates through each training pass based on #epochs & GPU/CPU
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(dataloaders_train):

            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) #check?
                
            
     # Forward and backward passes
            optimizer.zero_grad() # zero's out the gradient, otherwise will keep adding
            output = model.forward(inputs) # Forward propogation
            loss = criterion(output, labels) # Calculates loss
            loss.backward() # Calculates gradient
            optimizer.step() # Updates weights based on gradient & learning rate
            running_loss += loss.item()

            
            validation_loss, accuracy = validate(model, criterion, validation_loader)

            print("Epoch: {}/{} ".format(epoch+1, epochs),
                    "Training Loss: {:.3f} ".format(running_loss),
                    "Validation Loss: {:.3f} ".format(validation_loss),
                    "Validation Accuracy: {:.3f}".format(accuracy))
    


# In[12]:


# def train_model(model,epoches, learning_trate, criteion,)

# indicator=nn.NLLLoss() 
# optmizer=optim.Adam(model.classifier.parameters(),lr=0.002)
# model.to(device)

# epoch=15
# for e in range(epoch):
#     r_loss=0
#     for images,labels in dataloaders_train:
#         images, labels = images.to(device), labels.to(device)
        
#         optmizer.zero_grad()
#         output=model(images)
#         loss=indicator(output,labels)
#         loss.backward()
#         optmizer.step()
#         r_loss += loss.item()
    
#     else:
#         loss_test=0
#         model.eval()
#         with torch.no_grad():
            
#             for images,labels in dataloaders_test:
#                 images,labels=images.to(device),labels.to(device)
#                 output_t=model(images)
#                 t_loss=indicator(output_t,labels)
#                 loss_test+=t_loss.item()
#                 ps=torch.exp(output_t)
#                 top_p,top_class=ps.topk(1,dim=1)
            
#                 equals = top_class == labels.view(*top_class.shape)
#                 accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
                
            
#         print('running loss is {}, test loss is{}, accuracy is {}'\
#               .format(r_loss/len(dataloaders_train), loss_test/len(dataloaders_test), accuracy))
# #         model.train()      
        


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[13]:


epochs = 4
learning_rate = 0.001
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
train(model, epochs, learning_rate, criterion, optimizer, dataloaders_train, dataloaders_valid,'GPU')


# In[14]:


# loss_test=0
# for images,labels in dataloaders_test:
#     images,labels=images.to(device),labels.to(device)
     
       
#     output=model(images)
#     loss_t = indicator(output,labels)
#     ps=torch.exp(output)  
#     top_p,top_class=ps.topk(1,dim=1)
#     loss_test+=loss_t
#     equals = top_class == labels.view(*top_class.shape)
#     accuracy = torch.mean(equals.type(torch.FloatTensor)).item()

# print("accuracy is {}".format(accuracy))
# #output=model(im)


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[15]:


model.class_to_idx =image_datasets_train.class_to_idx
# model.cpu()
torch.save({'arch': 'vgg19',
            'state_dict': model.state_dict(), # Holds all the weights and biases
            'class_to_idx': model.class_to_idx},
            'ashlaki_R2.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[16]:
