import torch
from torch import nn
import pytorch_lightning as pl
from typing import Tuple


# Helper to shortcut for residual blocks, don't do anything
def noop(x: torch.Tensor) -> torch.Tensor:
    return x

class ReLU(nn.Module):
    """ Semi-Periodic Activation Function
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(0.0, x)

class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int] = [7, 5, 3], bottleneck: bool = True) -> None:
        """ Inception Module to apply parallel convolution on time series.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Tuple[int, int, int], optional): Number of kernels. Defaults to [7, 5, 3].
            bottleneck (bool, optional): If apply a bottleneck layer to convert the number of channels to the correct shape. Defaults to True.
        """
        super().__init__()
        
        self.kernel_sizes = kernel_size
        # Only apply bottleneck if the input channels number is bigger than 1
        bottleneck = bottleneck if in_channels > 1 else False
        self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False, padding='same') if bottleneck else noop
        # Calculate and apply convolutions for each kernel size
        self.convolutions = nn.ModuleList([
            nn.Conv1d(out_channels if bottleneck else in_channels, out_channels, kernel_size=k, padding='same', bias=False) for k in self.kernel_sizes
        ])
        # Max Convolutional Pooling layer
        self.maxconv = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, out_channels, 1, bias=False, padding='same')])
        self.batchnorm = nn.BatchNorm1d(out_channels * 4)
        self.activation = ReLU() # We just have to modify this activation between ReLU and LeakySineLU
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x
        x = self.bottleneck(x)
        # Conv1, Conv2, Conv3, MaxConv
        x = torch.cat([conv(x) for conv in self.convolutions] + [self.maxconv(x_)], dim=1)
        return self.activation(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = True, depth: int = 6) -> None:
        super().__init__()
        self.residual = residual
        self.depth = depth
        
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            # Build each inception module
            self.inception.append(InceptionModule(
                in_channels=(in_channels if d == 0 else out_channels * 4), out_channels=out_channels,
            ))
            if self.residual and d % 3 == 2:
                c_in, c_out = in_channels if d == 2 else out_channels * 4, out_channels * 4
                self.shortcut.append(
                    nn.BatchNorm1d(c_in) if c_in == c_out else nn.Sequential(*[nn.Conv1d(c_in, c_out, kernel_size=1, padding='same'), nn.BatchNorm1d(c_out)])
                )
        self.activation = ReLU() # We just have to modify this activation between ReLU and LeakySineLU
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
               res = x = self.activation(x + self.shortcut[d // 3](res))
        return x


class InceptionTime(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, sequence_len: int, class_num: int) -> None:
        super().__init__()
        self.inception_block = InceptionBlock(in_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels * 4, class_num)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception_block(x)
        x = torch.mean(x, dim=-1)
        return self.relu(self.fc(x))


inception = InceptionTime(in_channels=1, hidden_channels=32, sequence_len=120, class_num=2)


def train(model, optimizer, train_loader) -> float:
    model.train()
    batch_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda', non_blocking = True), labels.to('cuda', non_blocking = True)
        y_hat = model(inputs)
        loss = nn.CrossEntropyLoss(y_hat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss += float(loss)
    train_loss = batch_loss / len(train_loader)
    return train_loss
    
def test(model, test_loader) -> tuple([float, float]):
    model.eval()
    batch_correct = 0
    batch_loss = 0.0
    for inputs, labels in test_loader:
        model.eval()
        inputs, labels = inputs.to('cuda', non_blocking = True), labels.to('cuda', non_blocking = True)
        with torch.no_grad():
            y_hat = model(inputs)
            loss = nn.CrossEntropyLoss(y_hat, labels)
        batch_loss += float(loss)
        batch_correct += (torch.argmax(y_hat,1) == labels).float().sum()
    test_loss = batch_loss/len(test_loader)
    test_acc = batch_correct/len(test_loader.dataset)

    return test_loss, test_acc


def train_n_epochs(model, optimizer, train_loader, val_loader, n_epochs):
    hist_loss_train = []
    hist_loss_validation = []

    for epoch in range(n_epochs):
        train_loss = train(model, optimizer, train_loader)
        val_loss, val_acc = test(model, optimizer, val_loader)

        hist_loss_train.append(train_loss)
        hist_loss_validation.append(val_loss)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.2f} Validation Loss: {val_loss:.2f} Validation Accuracy: {val_acc:.2f}")

    return hist_loss_train, hist_loss_validation