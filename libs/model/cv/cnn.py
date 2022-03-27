import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.softmax(self.linear_2(x))
        return x


class CNN(torch.nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.softmax(self.linear_2(x))
        return x


class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.dropout_2(x)
        x = self.softmax(self.linear_2(x))
        return x


class Cnn1D(nn.Module):
    def __init__(self, n_classes):
        super(Cnn1D, self).__init__()
        self.n_classes = n_classes

        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.drop = nn.Dropout(p=0.6)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.lin3 = nn.Linear(256, 100)
        self.lin4 = nn.Linear(100, self.n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, x.shape[1])

        # Convolutional layers with ReLu activations
        a = torch.relu(self.conv1(x))
        a = torch.relu(self.conv2(a))
        a = self.drop(a)
        a = self.pool(a)
        # Fully connected layers
        a = a.view((batch_size, -1))
        a = torch.relu(self.lin3(a))
        a = torch.relu(self.lin4(a))

        return a


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Cifar10Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN32(torch.nn.Module):
    def __init__(self, channels, output_dim):
        super(CNN32, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(channels, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.softmax(self.linear_2(x))
        return x


class CNN_OriginalFedAvg_umdaa02fd(torch.nn.Module):

    def __init__(self, classes=44):
        super(CNN_OriginalFedAvg_umdaa02fd, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 128, kernel_size=(5, 5), padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(12_544, 512)
        # self.linear_1 = nn.Linear(49,152, 512)
        self.linear_2 = nn.Linear(512, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 128, 128)
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.softmax(self.linear_2(x))
        return x


class CNN_umdaa02fd_test2(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN_umdaa02fd_test2, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(50176, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.reshape(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


class CNN_OriginalFedAvg_fall(torch.nn.Module):

    def __init__(self, classes=44, features=11):
        super(CNN_OriginalFedAvg_fall, self).__init__()
        self.conv2d_1 = torch.nn.Conv1d(3, features, kernel_size=(5, 5), padding=2)
        self.max_pooling = nn.MaxPool1d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv1d(features, features * 2, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(features * 2 * 5 * 5 * 2 - features, 128)
        # self.linear_1 = nn.Linear(49,152, 512)
        self.linear_2 = nn.Linear(128, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, 128, 128)
        # x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.softmax(self.linear_2(x))
        return x
