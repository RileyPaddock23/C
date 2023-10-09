#Setup with libraries for math, displays, and ML library

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

#Defining a Function to preprocess the image
#We take each image and then normalize their rgb vals to 0 to 1
#This is a torch Tensor object
#Then we normalize using a z-score
#(each data point is represented by the number of standard deviations from the mean)
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

#Load MNIST handwritten digits data set (60000 train, 10000 test) and apply our preprocessing
#Our "loader" shuffles the dataset and then breaks it into batches of 64 items
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

#At this point the "shape" of our data is [64,1,28,28]
#This is because we have a set of 64 images and each is 28 pixels by 28 pixels
#We can look at one of the images:

#This is an iterable of our training data
dataiter = iter(trainloader)
#This is the first object in our grouped training data
images, labels = next(dataiter)
#This looks at the first image, converts it from a Tensor to a numpy array, 
#then removes any 1-dimensional part
print(images.shape)
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
# plt.show()

#Now we are going to build our neural network
#In practice they are designed very differently
#In this case the design is very simple

#Our input size is 28*28 for all the vals
input_size = 784
#We have 2 hidden layers which I will explain
hidden_sizes = [128, 64]
output_size = 10

#tensorflow lets us put together these layers in a CNN
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
#Each layer has a different activation function
#The input is linear, the hidden layers are ReLU
#(linear for pos values, 0 for negative ones)
#The last layer uses a LogSoftmax
#This normalizes our results and then makes them integers


#Now we need to define how we will train our network

#This defines how we score a certain set of weights
#for out network, its a normal loss function that 
#expects log inputs like LogSoftmax
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

#This initializes all the weights to a first guess
#based on the math behind the neural networks
loss.backward()

#TRAINING!
#This is a tool provided by pytorch to optimize our model
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
#We start with 15 iterations of gradient descent
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    guess_prob = max(ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    return guess_prob

#TESTING!!
worst_guesses = [2 for _ in range(10)]
correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):

    #Re-shape our data for testing
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    #Take our model and test its prediction
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    all_count += 1
    if(true_label == pred_label):
      correct_count += 1
    else:
        guess_prob = view_classify(img,ps)
        if guess_prob < worst_guesses[true_label]:
            worst_guesses[true_label] = guess_prob
            plt.savefig("images/"+str(true_label)+"_misscalc.png")
        plt.close()

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

