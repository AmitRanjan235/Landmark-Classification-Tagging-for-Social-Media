import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super(MyModel,self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        # Define the feature extractor layers
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        
        
        # Max pooling layer (divides the image by a factor of 2)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 256, 512)
        self.fc2 = nn.Linear(512, num_classes) # num_classes = 50
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        # Batch norm
        self.batch_norm2d_1 = nn.BatchNorm2d(16)
        self.batch_norm2d_2 = nn.BatchNorm2d(32)
        self.batch_norm2d_3 = nn.BatchNorm2d(64)
        self.batch_norm2d_4 = nn.BatchNorm2d(128)
        self.batch_norm1d = nn.BatchNorm1d(512)
        

    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.batch_norm2d_1(x)
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = self.batch_norm2d_2(x)
        x = self.pool(self.leaky_relu(self.conv3(x)))
        x = self.batch_norm2d_3(x)
        x = self.pool(self.leaky_relu(self.conv4(x)))
        x = self.batch_norm2d_4(x)
        x = self.pool(self.leaky_relu(self.conv5(x)))
        
        # flatten the image
        x = x.view(-1, 7 * 7 * 256)
        
        # dropout layer
        x = self.dropout(x)
        
        # 1st hidden layer
        x = self.leaky_relu(self.fc1(x))
        
        x = self.batch_norm1d(x)
        
        # dropout layer
        x = self.dropout(x)
        
        # final layer
        x = self.fc2(x)
        
        return x





   

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
