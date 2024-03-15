import gguf
import torch
import torch.nn as nn

# Define the model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

model = CNN()
model.load_state_dict(torch.load("mnist_model.pth"))

g = gguf.GGUFWriter("mnist-cnn-model-pt.gguf", "mnist-cnn")
g.add_tensor("kernel1", model.conv1.weight.data.numpy())
g.add_tensor("bias1", model.conv1.bias.data.numpy())
g.add_tensor("kernel2", model.conv2.weight.data.numpy())
g.add_tensor("bias2", model.conv2.bias.data.numpy())
g.add_tensor("dense_w", model.fc1.weight.data.numpy())
g.add_tensor("dense_b", model.fc1.bias.data.numpy())
g.write_header_to_file()
g.write_kv_data_to_file()
g.write_tensors_to_file()
g.close()
