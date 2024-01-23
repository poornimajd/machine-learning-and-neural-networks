import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
'''
In this file you will write end-to-end code to train a neural network to categorize fashion-mnist data
'''


'''
PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted.
'''

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])  # Use transforms to convert images to tensors and normalize them
batch_size = 64

'''
PART 2:
Load the dataset. Make sure to utilize the transform and batch_size you wrote in the last section.
'''

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

'''
PART 3:
Design a multi layer perceptron. Since this is a purely Feedforward network, you mustn't use any convolutional layers
Do not directly import or copy any existing models.
'''

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
net = Net()
print(net)
'''
PART 4:
Choose a good loss function and optimizer
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

'''
PART 5:
Train your model!
'''

num_epochs = 5

loss_list=[]
count=0
iteration_list=[]

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if not (count % 50):
          loss_list.append(loss.item())
          iteration_list.append(count)
        count+=1

    print(f"Epoch {epoch+1}, Training loss: {running_loss / len(trainloader)}")
    print(f"Training loss: {running_loss}")

print('Finished Training')


plt.plot(iteration_list, loss_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Loss")
plt.title("Iterations vs Loss")
plt.show()

'''
PART 6:
Evalute your model! Accuracy should be greater or equal to 80%

'''

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

'''
PART 7:
Check the written portion. You need to generate some plots. 
'''
def view_classify(img, ps, actual,version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    predicted = torch.max(ps, 1)[1]
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    

    # print(output_mapping[int(labels[10])])
    ax1.set_title('predicted: {} actual: {}'.format(str(output_mapping[int(predicted)]),str(actual)))
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()



output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat",
                 5: "Sandal",
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }



# Testing out the network
dataiter = iter(testloader)
images, labels = next(dataiter)


#correct prediction example

img = images[10]

# # Convert 2D image to 1D vector
img = img.resize_(1, 28*28)

# # Calculate the class probabilites (log softmax) for img
ps = torch.exp(net(img))

# # Plot the image and probabilites

view_classify(img, ps, output_mapping[int(labels[10])],version='Fashion')

#all the wrong samples
for i in range(len(images)):

    img = images[i]

    # # Convert 2D image to 1D vector
    img = img.resize_(1, 28*28)

    # # Calculate the class probabilites (log softmax) for img
    ps = torch.exp(net(img))

    # # Plot the image and probabilites


    predicted = torch.max(ps, 1)[1]

    if int(labels[i])==int(predicted):
      print("true")

    else:
      view_classify(img, ps, output_mapping[int(labels[i])],version='Fashion')
