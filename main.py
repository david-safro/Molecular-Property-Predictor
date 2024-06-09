import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, global_mean_pool

# Load the QM9 dataset
dataset = QM9(root='data/QM9')

# Split the dataset into training, validation, and test sets
train_dataset = dataset[:10000]
val_dataset = dataset[10000:11000]
test_dataset = dataset[11000:12000]

# Create data loaders for each split
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the GNN Model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Global mean pooling
        x = self.lin(x)
        return x

# Training and Testing Functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            error += F.mse_loss(out, data.y).item()
    return error / len(loader)

# Training Loop
for epoch in range(1, 101):
    train()
    val_error = test(val_loader)
    print(f'Epoch: {epoch:03d}, Validation Error: {val_error:.4f}')

test_error = test(test_loader)
print(f'Test Error: {test_error:.4f}')

# Testing Specific Elements
def test_specific_elements(elements):
    model.eval()
    with torch.no_grad():
        for idx in elements:
            data = dataset[idx].to(device)
            out = model(data)
            actual = data.y
            print(f'Molecule {idx}:')
            print(f'Predicted: {out.cpu().numpy()}')
            print(f'Actual: {actual.cpu().numpy()}\n')

# Test specific elements (change indices as needed)
specific_elements = [11005, 11006, 11007]  # Example molecule indices
test_specific_elements(specific_elements)
