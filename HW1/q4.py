"""
q4 Graph Neural Networks Tutorial - PyCharm Implementation
Reproduces results from UvA DL Tutorial 7
"""

import os
import json
import math
import numpy as np
import time
import urllib.request
from urllib.error import HTTPError

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    print("Installing PyTorch Lightning...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-lightning>=1.4"])
    import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# PyTorch Geometric
try:
    import torch_geometric
except ModuleNotFoundError:
    print("Installing PyTorch Geometric...")
    import subprocess
    import sys

    # Get PyTorch and CUDA versions
    TORCH = torch.__version__.split('+')[0]
    CUDA = 'cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'

    # Install PyTorch Geometric dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-scatter", "-f",
                           f"https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-sparse", "-f",
                           f"https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-cluster", "-f",
                           f"https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-spline-conv", "-f",
                           f"https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])

    import torch_geometric

import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data

# Set paths
DATASET_PATH = "./data"
CHECKPOINT_PATH = "./saved_models/tutorial7"

# Create directories if they don't exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Set random seed for reproducibility
pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


# Download pretrained models
def download_pretrained_models():
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial7/"
    pretrained_files = ["NodeLevelMLP.ckpt", "NodeLevelGNN.ckpt", "GraphLevelGraphConv.ckpt"]

    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
                print(f"Downloaded {file_name}")
            except HTTPError as e:
                print(f"Could not download {file_name}: {e}")


# ============================================================================
# CUSTOM LAYER IMPLEMENTATIONS
# ============================================================================

class GCNLayer(nn.Module):
    """Graph Convolutional Network Layer"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats


class GATLayer(nn.Module):
    """Graph Attention Network Layer"""

    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0
            c_out = c_out // num_heads

        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out))
        self.leakyrelu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        edges = adj_matrix.nonzero(as_tuple=False)
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1)

        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats


# ============================================================================
# MODELS USING PYTORCH GEOMETRIC
# ============================================================================

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}


class GNNModel(nn.Module):
    """General GNN model using PyTorch Geometric layers"""

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class MLPModel(nn.Module):
    """MLP baseline model"""

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


# ============================================================================
# NODE-LEVEL CLASSIFICATION
# ============================================================================

class NodeLevelGNN(pl.LightningModule):
    """PyTorch Lightning module for node-level tasks"""

    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)


def train_node_classifier(model_name, dataset, **model_kwargs):
    """Train node-level classifier"""
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=200,
                         enable_progress_bar=False)
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"NodeLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model for {model_name}, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        print(f"Training {model_name} from scratch...")
        pl.seed_everything(42)
        model = NodeLevelGNN(model_name=model_name,
                             c_in=dataset.num_node_features,
                             c_out=dataset.num_classes,
                             **model_kwargs)
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    test_result = trainer.test(model, node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_result[0]['test_acc']}
    return model, result


# ============================================================================
# GRAPH-LEVEL CLASSIFICATION
# ============================================================================

class GraphGNNModel(nn.Module):
    """GNN model for graph-level tasks"""

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        super().__init__()
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden,
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = self.head(x)
        return x


class GraphLevelGNN(pl.LightningModule):
    """PyTorch Lightning module for graph-level tasks"""

    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.0)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)


def train_graph_classifier(model_name, train_loader, val_loader, test_loader,
                           num_node_features, num_classes, **model_kwargs):
    """Train graph-level classifier"""
    pl.seed_everything(42)

    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=500,
                         enable_progress_bar=True)
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model for {model_name}, loading...")
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        print(f"Training {model_name} from scratch...")
        pl.seed_everything(42)
        model = GraphLevelGNN(c_in=num_node_features,
                              c_out=1 if num_classes == 2 else num_classes,
                              **model_kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]['test_acc'],
              "train": train_result[0]['test_acc']}
    return model, result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_results(result_dict):
    """Print results in a formatted way"""
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0 * result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy: {(100.0 * result_dict['val']):4.2f}%")
    print(f"Test accuracy: {(100.0 * result_dict['test']):4.2f}%")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to reproduce all results"""

    print("=" * 60)
    print("GRAPH NEURAL NETWORKS TUTORIAL - REPRODUCING RESULTS")
    print("=" * 60)

    # Download pretrained models if available
    print("\n1. Checking for pretrained models...")
    download_pretrained_models()

    # ========================================
    # PART 1: NODE-LEVEL CLASSIFICATION (CORA)
    # ========================================
    print("\n" + "=" * 60)
    print("PART 1: NODE-LEVEL CLASSIFICATION ON CORA DATASET")
    print("=" * 60)

    # Load Cora dataset
    print("\nLoading Cora dataset...")
    cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")
    print(f"Dataset info: {cora_dataset[0]}")
    print(f"Number of nodes: {cora_dataset[0].x.shape[0]}")
    print(f"Number of features: {cora_dataset[0].x.shape[1]}")
    print(f"Number of classes: {cora_dataset.num_classes}")

    # Train MLP baseline
    print("\n" + "-" * 40)
    print("Training MLP baseline...")
    print("-" * 40)
    node_mlp_model, node_mlp_result = train_node_classifier(
        model_name="MLP",
        dataset=cora_dataset,
        c_hidden=16,
        num_layers=2,
        dp_rate=0.1
    )
    print("\nMLP Results:")
    print_results(node_mlp_result)

    # Train GNN
    print("\n" + "-" * 40)
    print("Training GNN (GCN)...")
    print("-" * 40)
    node_gnn_model, node_gnn_result = train_node_classifier(
        model_name="GNN",
        layer_name="GCN",
        dataset=cora_dataset,
        c_hidden=16,
        num_layers=2,
        dp_rate=0.1
    )
    print("\nGNN Results:")
    print_results(node_gnn_result)

    # ========================================
    # PART 2: GRAPH-LEVEL CLASSIFICATION (MUTAG)
    # ========================================
    print("\n" + "=" * 60)
    print("PART 2: GRAPH-LEVEL CLASSIFICATION ON MUTAG DATASET")
    print("=" * 60)

    # Load MUTAG dataset
    print("\nLoading MUTAG dataset...")
    tu_dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG")
    print(f"Dataset info: {tu_dataset.data}")
    print(f"Number of graphs: {len(tu_dataset)}")
    print(f"Average label: {tu_dataset.data.y.float().mean().item():4.2f}")

    # Split dataset
    torch.manual_seed(42)
    tu_dataset.shuffle()
    train_dataset = tu_dataset[:150]
    test_dataset = tu_dataset[150:]

    # Create data loaders
    graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    graph_val_loader = geom_data.DataLoader(test_dataset, batch_size=64)
    graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=64)

    # Train GraphConv model
    print("\n" + "-" * 40)
    print("Training Graph-level GNN (GraphConv)...")
    print("-" * 40)
    graph_model, graph_result = train_graph_classifier(
        model_name="GraphConv",
        train_loader=graph_train_loader,
        val_loader=graph_val_loader,
        test_loader=graph_test_loader,
        num_node_features=tu_dataset.num_node_features,
        num_classes=tu_dataset.num_classes,
        c_hidden=256,
        layer_name="GraphConv",
        num_layers=3,
        dp_rate_linear=0.5,
        dp_rate=0.0
    )
    print("\nGraph Classification Results:")
    print(f"Train performance: {100.0 * graph_result['train']:4.2f}%")
    print(f"Test performance: {100.0 * graph_result['test']:4.2f}%")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL RESULTS")
    print("=" * 60)

    print("\n1. Node-level Classification (Cora):")
    print("   MLP baseline:")
    print(f"      - Test accuracy: {100.0 * node_mlp_result['test']:4.2f}%")
    print("   GNN (GCN):")
    print(f"      - Test accuracy: {100.0 * node_gnn_result['test']:4.2f}%")

    print("\n2. Graph-level Classification (MUTAG):")
    print("   GraphConv:")
    print(f"      - Train accuracy: {100.0 * graph_result['train']:4.2f}%")
    print(f"      - Test accuracy: {100.0 * graph_result['test']:4.2f}%")

    print("\n" + "=" * 60)
    print("REPRODUCTION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()