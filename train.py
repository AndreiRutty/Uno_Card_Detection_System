import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=60),  # Randomly rotate images within 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Adjust brightness, contrast, saturation, and hue
    transforms.Resize((254, 254)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = 'dataset'
train_dataset = ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

num_classes = len(train_dataset.classes)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # Second convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer

        # Calculate the size after conv layers and pooling to define fc1 input size
        self.fc1 = nn.Linear(32 * 62 * 62, 128)  # Update based on the output size after conv and pooling
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Apply pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Apply pooling
        print("Shape after conv layers:", x.shape)  # Debugging line
        x = x.view(-1, 32 * 62 * 62)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    model = CNN(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), 'uno_model.pt')

    print("Training Completed")
