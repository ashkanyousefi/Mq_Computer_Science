
    # Region Imports ***
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
#endregion


# ************* Start Imports 

def load_data()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

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

    image_datasets_train = datasets.ImageFolder( train_dir, transform=data_transforms_train)
    image_datasets_valid = datasets.ImageFolder( valid_dir, transform=data_transforms_test)
    image_datasets_test = datasets.ImageFolder( test_dir, transform=data_transforms_test)

    dataloaders_train = torch.utils.data.DataLoader(image_datasets_train,batch_size=64,shuffle=True)
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid,batch_size=64)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test,batch_size=64)

    class_to_idx = image_datasets_train.class_to_idx

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

# ************* End Imports 


# ************* Start Creating Model 

def create_model(model_name)
    if (model_name.lower()=="vgg16"):
        model =models.vgg16(pretrained=True)
    else if (model_name.lower()=="vgg19"):
        model=models.vgg19(pretrained=True)

    print(model)
    for param in model.parameters():
        param.requires_grad=False

    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, 4096)), # First layer
                        ('relu', nn.ReLU()), # Apply activation function
                        ('fc2', nn.Linear(4096, 102)), # Output layer
                        ('output', nn.LogSoftmax(dim=1)) # Apply loss function
                        ]))

    model.classifier = classifier


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



def train(model, epochs_n, learning_rate, criterion, optimizer, training_loader, validation_loader,comp_mode):
    epochs = epochs_n
    model.train()
    use_gpu = False
    
    if comp_mode.lower()=="gpu":
        use_gpu = True
        model.cuda()
    else:
        model.cpu()

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(dataloaders_train):

            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) #check?
                
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
    

#epochs = 0
learning_rate = 0.001
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
train(model, epochs, learning_rate, criterion, optimizer, dataloaders_train, dataloaders_valid,'GPU')


# In[ ]:





# In[14]:


loss_test=0
use_gpu = False
comp_mode='GPU'
criterion = nn.NLLLoss()  
# Check to see whether GPU is available
if comp_mode.lower()=="gpu":
    use_gpu = True
    model.cuda()
else:
    model.cpu()

for images,labels in dataloaders_test:
    
    if use_gpu:
        images = Variable(images.float().cuda())
        labels = Variable(labels.long().cuda()) 
    else:
        images = Variable(images)
        labels = Variable(labels)

    #images,labels=images.to(device),labels.to(device)
     
       
    output=model(images)
    loss_t = criterion(output,labels)
    ps=torch.exp(output)  
    top_p,top_class=ps.topk(1,dim=1)
    loss_test+=loss_t
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()

print("accuracy is {}".format(accuracy))
#output=model(im)


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
# torch.save({'arch': 'vgg19',
#             'state_dict': model.state_dict(), # Holds all the weights and biases
#             'class_to_idx': model.class_to_idx},
#             'ashlaki_R2.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[16]:


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
#     model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    
#     classifier = nn.Sequential(OrderedDict([
#                           ('fc1', nn.Linear(25088, 4096)),
#                           ('relu', nn.ReLU()),
#                           ('fc2', nn.Linear(4096, 102)),
#                           ('output', nn.LogSoftmax(dim=1))
#                           ]))
    
#     model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


# In[17]:


model=load_model('ashlaki_R2.pth')
model


# In[18]:


# model.load_state_dict(torch.load('ashlaki.pth'))


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# 

# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[19]:


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img=Image.open(image_path)
    
#     img_pro=transforms.Compose([transforms.Resize(256),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406], 
#                                                            [0.229, 0.224, 0.225])])
#     img_tensor = img_pro(img)
#     return img_tensor
    img=img.resize((256,256))
    value=0.5*(256-224)
    img=img.crop((value,value,256-value,256-value))
    img=np.array(img)/255
    
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    img=(img-mean)/std
    
    return img.transpose(2,0,1)


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[20]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
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


# In[21]:


imshow(process_image('./flowers/test/10/image_07090.jpg'))


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[22]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    else:
        model.cpu()
        
    img=process_image(image_path)
#     img.unsqueeze_(0)
#     img.unsqueez_(0)
    img = torch.from_numpy(np.array([img])).float()

    
    # The image becomes the input
#     img = Variable(img)
    if cuda:
        img = img.cuda()

    output=model.forward(img)
    log_out=torch.exp(output)
    top_p,top_class = torch.topk(log_out, topk)
    idx_to_class={}
    for key,value in model.class_to_idx.items():
        idx_to_class[value]=key
    
    
    np_top_class=np.array(top_class[0])    
    top_class_name=[]
    for labels in np_top_class:
        top_class_name.append(int(idx_to_class[labels]))
    
    top_flowers=[cat_to_name[str(lab)] for lab in top_class_name]
    
    
    
    
    for x in top_class.tolist()[0]:
        if x>0:
            top_class_name.append(cat_to_name[str(x)])
        
    return top_p,top_class_name,top_flowers


# In[88]:


top_p,top_class,top_class_name=predict("./flowers/test/10/image_07090.jpg",model,topk=5)
print(top_p)
#print(top_class)
print(top_class_name)
f_list=[]
for item in top_p.tolist()[0]:
        f_list.append(item),
index=objects
plt.figure(figsize=(12,5))
plt.bar(index,f_list)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[ ]:




