import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


SIZE = 50   # TRAINING DATA IMG SIZE
LABELS = 2  # cat or dog

training_data = np.load("training_data_50.npy", allow_pickle=True)  # load pre-made dataset
print("The length of the training data" , len(training_data))


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,86,3)
        self.conv3 = nn.Conv2d(86,128,3)

        x = torch.randn(SIZE,SIZE).view(-1,1,SIZE,SIZE)

        self._to_linear = None
        self.activation(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening
        self.fc2 = nn.Linear(512, LABELS)

    def activation(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear == None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        x = self.activation(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = Net()
learning_rate = 0.001

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,SIZE,SIZE)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1   # save 10% of the data for test
val_size = int(len(X)*VAL_PCT)


train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print("Length of train_x : " ,len(train_X)) # !
print("Length of test_x : " , len(test_X))  # !
BATCH_SIZE = 100
EPOCHS = 2

for epoch in range(EPOCHS):
    if epoch == 0:
        print("starting the first epoch")
    else:
        print(EPOCHS - epoch , " Epoch(s) left..")
    
    for i in range(0, len(train_X), BATCH_SIZE):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,SIZE,SIZE)
        batch_y = train_y[i:i+BATCH_SIZE]
        
        if i % 1000 == 0 and i != 0:
            print(i, " images processed.")


        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print("The current loss value is : ", loss)

correct = 0
total = 0

with torch.no_grad():
    for i in range(len(test_X)):
        real_class = torch.argmax(test_y[i])
        output = net(test_X[i].view(-1,1,SIZE,SIZE))[0]
        predicted_class = torch.argmax(output)
        if predicted_class == real_class:
            correct += 1
        total += 1
        print("Cat == 0 & Dog == 1")
        print("Prediction : ", predicted_class)
        print("Reality : ", real_class)
        print(" ")

print("Total guesses: ", total)
print("Correct guesses: ", correct)
print("Accuracy of the guesses: ", round(correct/total,3))
