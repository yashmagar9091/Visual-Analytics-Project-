# Run: python model_training.py
# ==========================================
# TASK 1: MOLECULAR SOLUBILITY PREDICTION
# Standalone Training & Artifact Generation Script
# ==========================================

import copy
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so script runs cleanly in terminal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
BEST_LAYERS = 4  # Upgraded final model depth (4-layer GraphSAGE)

TARGET = "measured log solubility in mols per litre"
FEATURES = [
    "Minimum Degree",
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Polar Surface Area",
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# MOLECULAR GRAPH CONVERSION
# ==========================================

def smiles_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
        ])

    x = torch.tensor(atom_features, dtype=torch.float)

    # Bond connections (undirected)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([target], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


def create_graph_dataset(df, X_split, y_split):
    graphs = []
    for idx in X_split.index:
        smiles = df.loc[idx, "smiles"]
        target = y_split.loc[idx]
        graph = smiles_to_graph(smiles, target)
        if graph is not None:
            graphs.append(graph)
    return graphs


# ==========================================
# GNN ARCHITECTURES
# ==========================================

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(5, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 32)
        self.lin2 = Linear(32, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).view(-1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(5, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 32)
        self.lin2 = Linear(32, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).view(-1)


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(5, hidden_channels, heads=1, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
        self.lin1 = Linear(hidden_channels, 32)
        self.lin2 = Linear(32, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).view(-1)


class GraphSAGEPooling(torch.nn.Module):
    def __init__(self, pooling="mean", hidden_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(5, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 32)
        self.lin2 = Linear(32, 1)
        self.dropout = dropout
        self.pooling = pooling

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).view(-1)


class GraphSAGEFlexible(torch.nn.Module):
    def __init__(self, num_layers=2, hidden_channels=64, dropout=0.2, pooling="sum"):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(5, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.lin1 = Linear(hidden_channels, 32)
        self.lin2 = Linear(32, 1)
        self.dropout = dropout
        self.pooling = pooling

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).view(-1)


# ==========================================
# TRAINING & EVALUATION HELPERS
# ==========================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate_metrics(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            actual.extend(batch.y.view(-1).cpu().numpy())
            predicted.extend(predictions.cpu().numpy())

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    r2 = float(r2_score(actual, predicted))
    return rmse, mae, r2, np.array(actual), np.array(predicted)


def fit_model(model, train_loader, val_loader, lr, epochs, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_state, best_epoch, best_val_loss, train_losses, val_losses


def get_split_predictions(df, model, graphs, split_indices, split_name, device):
    model.eval()
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.extend(model(batch).cpu().numpy())

    sub_df = df.loc[split_indices].copy()
    sub_df["Molecule_Index"] = sub_df.index
    sub_df["Split"] = split_name
    sub_df["Measured_logS"] = sub_df[TARGET]
    sub_df["Predicted_logS"] = preds
    sub_df["Error (Pred - Actual)"] = sub_df["Predicted_logS"] - sub_df["Measured_logS"]
    sub_df["Absolute_Error"] = sub_df["Error (Pred - Actual)"].abs()
    return sub_df


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():
    set_seed(SEED)

    # Paths (anchored to script directory, matching Task1_dashboard.py)
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "Data" / "delaney-processed.csv"
    artifact_dir = project_dir / "artifacts"
    model_dir = artifact_dir / "models"
    results_dir = artifact_dir / "results"
    checkpoint_dir = artifact_dir / "checkpoints"

    for folder in [model_dir, results_dir, checkpoint_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()

    print(f"Project directory: {project_dir}")
    print(f"Dataset path:      {data_path}")
    print(f"Device:            {device}")

    # 1. Load dataset
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Dataset loaded: shape={df.shape}, missing={df.isna().sum().sum()}, duplicates={df.duplicated().sum()}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # 2. Primary 70/15/15 split (Seed = 42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED)

    train_graphs = create_graph_dataset(df, X_train, y_train)
    val_graphs = create_graph_dataset(df, X_val, y_val)
    test_graphs = create_graph_dataset(df, X_test, y_test)
    print(f"Graphs created -> Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Mean Baseline vs. Initial GCN Baseline
    print("\n--- Step 1: Mean Baseline & GCN Baseline ---")
    set_seed(SEED)
    gcn_model = GCN(hidden_channels=64, dropout=0.2).to(device)
    gcn_state, gcn_best_epoch, gcn_best_loss, gcn_train_losses, gcn_val_losses = fit_model(
        gcn_model, train_loader, val_loader, LEARNING_RATE, EPOCHS, criterion, device
    )
    torch.save({
        "model_state_dict": gcn_state,
        "best_epoch": gcn_best_epoch,
        "best_val_loss": gcn_best_loss,
        "hidden_channels": 64,
        "dropout": 0.2,
        "learning_rate": LEARNING_RATE,
    }, checkpoint_dir / "gcn_baseline.pt")

    # Save GCN loss curve image
    plt.figure(figsize=(9, 5))
    plt.plot(gcn_train_losses, label="Training Loss")
    plt.plot(gcn_val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("GCN Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "gcn_training_curve.png", dpi=150)
    plt.close()

    gcn_val_rmse, gcn_val_mae, _, _, _ = evaluate_metrics(gcn_model, val_loader, device)
    pd.DataFrame([{
        "Model": "GCN",
        "Architecture": "2-layer GCN",
        "Hidden Channels": 64,
        "Pooling": "Global Mean",
        "Dropout": 0.2,
        "Learning Rate": LEARNING_RATE,
        "Best Epoch": gcn_best_epoch,
        "Validation RMSE": gcn_val_rmse,
        "Validation MAE": gcn_val_mae,
    }]).to_csv(results_dir / "gcn_baseline_results.csv", index=False)

    baseline_preds = np.full(len(y_val), y_train.mean())
    baseline_rmse = float(np.sqrt(mean_squared_error(y_val, baseline_preds)))
    baseline_mae = float(mean_absolute_error(y_val, baseline_preds))

    comparison_df = pd.DataFrame([
        {"Model": "Mean Baseline", "Validation RMSE": baseline_rmse, "Validation MAE": baseline_mae},
        {"Model": "GCN", "Validation RMSE": gcn_val_rmse, "Validation MAE": gcn_val_mae},
    ])
    comparison_df.to_csv(results_dir / "baseline_comparison.csv", index=False)
    print(comparison_df.to_string(index=False))

    # 4. Architecture Comparison (GCN vs. GraphSAGE vs. GAT)
    print("\n--- Step 2: Architecture Comparison (GCN vs. GraphSAGE vs. GAT) ---")
    set_seed(SEED)
    sage_model = GraphSAGE(hidden_channels=64, dropout=0.2).to(device)
    sage_state, sage_best_epoch, sage_best_loss, _, _ = fit_model(
        sage_model, train_loader, val_loader, LEARNING_RATE, EPOCHS, criterion, device
    )
    torch.save({
        "model_state_dict": sage_state,
        "best_epoch": sage_best_epoch,
        "best_val_loss": sage_best_loss,
        "hidden_channels": 64,
        "dropout": 0.2,
        "learning_rate": LEARNING_RATE,
    }, checkpoint_dir / "graphsage_baseline.pt")
    sage_val_rmse, sage_val_mae, _, _, _ = evaluate_metrics(sage_model, val_loader, device)

    set_seed(SEED)
    gat_model = GAT(hidden_channels=64, dropout=0.2).to(device)
    gat_state, gat_best_epoch, gat_best_loss, _, _ = fit_model(
        gat_model, train_loader, val_loader, LEARNING_RATE, EPOCHS, criterion, device
    )
    torch.save({
        "model_state_dict": gat_state,
        "best_epoch": gat_best_epoch,
        "best_val_loss": gat_best_loss,
        "hidden_channels": 64,
        "dropout": 0.2,
        "learning_rate": LEARNING_RATE,
    }, checkpoint_dir / "gat_baseline.pt")
    gat_val_rmse, gat_val_mae, _, _, _ = evaluate_metrics(gat_model, val_loader, device)

    arch_df = pd.DataFrame([
        {"Model": "GCN", "Validation RMSE": gcn_val_rmse, "Validation MAE": gcn_val_mae, "Best Epoch": gcn_best_epoch},
        {"Model": "GraphSAGE", "Validation RMSE": sage_val_rmse, "Validation MAE": sage_val_mae, "Best Epoch": sage_best_epoch},
        {"Model": "GAT", "Validation RMSE": gat_val_rmse, "Validation MAE": gat_val_mae, "Best Epoch": gat_best_epoch},
    ])
    arch_df.to_csv(results_dir / "gnn_architecture_comparison.csv", index=False)
    print(arch_df.to_string(index=False))

    # 5. Pooling Comparison (Mean vs. Max vs. Sum)
    print("\n--- Step 3: GraphSAGE Pooling Comparison ---")
    pooling_results = []
    sum_best_state = None
    for pooling_method in ["mean", "max", "sum"]:
        set_seed(SEED)
        p_model = GraphSAGEPooling(pooling=pooling_method, hidden_channels=64, dropout=0.2).to(device)
        p_state, p_epoch, _, _, _ = fit_model(
            p_model, train_loader, val_loader, LEARNING_RATE, EPOCHS, criterion, device
        )
        rmse, mae, r2, _, _ = evaluate_metrics(p_model, val_loader, device)
        pooling_results.append({
            "Pooling": pooling_method,
            "Best Epoch": p_epoch,
            "Validation RMSE": rmse,
            "Validation MAE": mae,
            "Validation R2": r2,
        })
        if pooling_method == "sum":
            sum_best_state = p_state

    pooling_df = pd.DataFrame(pooling_results)
    pooling_df.to_csv(results_dir / "graphsage_pooling_comparison.csv", index=False)
    print(pooling_df.to_string(index=False))

    sum_row = pooling_df[pooling_df["Pooling"] == "sum"].iloc[0]
    torch.save({
        "model_state_dict": sum_best_state,
        "architecture": "GraphSAGE",
        "pooling": "sum",
        "hidden_channels": 64,
        "dropout": 0.2,
        "best_epoch": int(sum_row["Best Epoch"]),
        "validation_rmse": float(sum_row["Validation RMSE"]),
        "validation_mae": float(sum_row["Validation MAE"]),
        "validation_r2": float(sum_row["Validation R2"]),
    }, checkpoint_dir / "graphsage_sum_best.pt")

    # 6. Hidden Dimension Comparison (64 vs. 128)
    print("\n--- Step 4: Hidden Dimension Comparison ---")
    hidden_results = []
    for hidden_size in [64, 128]:
        set_seed(SEED)
        h_model = GraphSAGEPooling(pooling="sum", hidden_channels=hidden_size, dropout=0.2).to(device)
        _, h_epoch, _, _, _ = fit_model(
            h_model, train_loader, val_loader, LEARNING_RATE, EPOCHS, criterion, device
        )
        rmse, mae, r2, _, _ = evaluate_metrics(h_model, val_loader, device)
        hidden_results.append({
            "Hidden Channels": hidden_size,
            "Best Epoch": h_epoch,
            "Validation RMSE": rmse,
            "Validation MAE": mae,
            "Validation R2": r2,
        })

    hidden_df = pd.DataFrame(hidden_results)
    hidden_df.to_csv(results_dir / "hidden_dimension_comparison.csv", index=False)
    print(hidden_df.to_string(index=False))

    # 7. Number of Layers Comparison (2 vs. 3 vs. 4)
    print("\n--- Step 5: Number of Layers Comparison ---")
    layer_results = []
    for num_layers in [2, 3, 4]:
        set_seed(SEED)
        l_model = GraphSAGEFlexible(
            num_layers=num_layers, hidden_channels=64, dropout=0.2, pooling="sum"
        ).to(device)
        _, l_epoch, _, _, _ = fit_model(
            l_model, train_loader, val_loader, LEARNING_RATE, EPOCHS, criterion, device
        )
        rmse, mae, r2, _, _ = evaluate_metrics(l_model, val_loader, device)
        layer_results.append({
            "Layers": num_layers,
            "Best Epoch": l_epoch,
            "Validation RMSE": rmse,
            "Validation MAE": mae,
            "Validation R2": r2,
        })

    layer_df = pd.DataFrame(layer_results)
    layer_df.to_csv(results_dir / "layer_comparison.csv", index=False)
    print(layer_df.to_string(index=False))

    # 8. Learning Rate Comparison (0.0005 vs. 0.001 vs. 0.005)
    print("\n--- Step 6: Learning Rate Comparison ---")
    lr_results = []
    for lr_val in [0.0005, 0.001, 0.005]:
        set_seed(SEED)
        lr_model = GraphSAGEPooling(pooling="sum", hidden_channels=64, dropout=0.2).to(device)
        _, lr_epoch, _, _, _ = fit_model(
            lr_model, train_loader, val_loader, lr_val, EPOCHS, criterion, device
        )
        rmse, mae, r2, _, _ = evaluate_metrics(lr_model, val_loader, device)
        lr_results.append({
            "Learning Rate": lr_val,
            "Best Epoch": lr_epoch,
            "Validation RMSE": rmse,
            "Validation MAE": mae,
            "Validation R2": r2,
        })

    lr_df = pd.DataFrame(lr_results)
    lr_df.to_csv(results_dir / "learning_rate_comparison.csv", index=False)
    print(lr_df.to_string(index=False))

    # 9. Dropout Comparison (0.0 vs. 0.2 vs. 0.4)
    print("\n--- Step 7: Dropout Comparison ---")
    dropout_results = []
    for drop_val in [0.0, 0.2, 0.4]:
        set_seed(SEED)
        d_model = GraphSAGEPooling(pooling="sum", hidden_channels=64, dropout=drop_val).to(device)
        _, d_epoch, _, _, _ = fit_model(
            d_model, train_loader, val_loader, LEARNING_RATE, EPOCHS, criterion, device
        )
        rmse, mae, r2, _, _ = evaluate_metrics(d_model, val_loader, device)
        dropout_results.append({
            "Dropout": drop_val,
            "Best Epoch": d_epoch,
            "Validation RMSE": rmse,
            "Validation MAE": mae,
            "Validation R2": r2,
        })

    dropout_df = pd.DataFrame(dropout_results)
    dropout_df.to_csv(results_dir / "dropout_comparison.csv", index=False)
    print(dropout_df.to_string(index=False))

    # 10. Final Reproducible Model Training (4-Layer GraphSAGE + Sum Pooling)
    print("\n--- Step 8: Final 4-Layer GraphSAGE Model Training ---")
    best_hidden = int(hidden_df.loc[hidden_df["Validation RMSE"].idxmin(), "Hidden Channels"])
    print(f"Selected Hidden Channels: {best_hidden} | Selected Layers: {BEST_LAYERS}")

    set_seed(SEED)
    final_model = GraphSAGEFlexible(
        num_layers=BEST_LAYERS,
        pooling="sum",
        hidden_channels=best_hidden,
        dropout=0.2,
    ).to(device)

    final_state, best_final_epoch, best_final_loss, final_train_losses, final_val_losses = fit_model(
        final_model, train_loader, val_loader, LEARNING_RATE, EPOCHS, criterion, device
    )

    torch.save({
        "model_state_dict": final_state,
        "architecture": "GraphSAGE",
        "num_layers": BEST_LAYERS,
        "pooling": "sum",
        "hidden_channels": best_hidden,
        "dropout": 0.2,
        "learning_rate": LEARNING_RATE,
        "best_epoch": best_final_epoch,
        "best_validation_mse": best_final_loss,
    }, checkpoint_dir / "final_graphsage_model.pt")

    # Evaluate final model on held-out Test set
    test_rmse, test_mae, test_r2, test_actual, test_predicted = evaluate_metrics(
        final_model, test_loader, device
    )
    errors = np.abs(test_actual - test_predicted)
    acc_05 = float(np.mean(errors <= 0.5) * 100)
    acc_10 = float(np.mean(errors <= 1.0) * 100)
    acc_15 = float(np.mean(errors <= 1.5) * 100)
    acc_20 = float(np.mean(errors <= 2.0) * 100)

    print("FINAL TEST RESULTS")
    print("==================")
    print(f"Best Epoch:         {best_final_epoch}")
    print(f"Best Val RMSE:      {np.sqrt(best_final_loss):.4f} log S")
    print(f"Test RMSE:          {test_rmse:.4f} log S")
    print(f"Test MAE:           {test_mae:.4f} log S")
    print(f"Test R2:            {test_r2:.4f}")
    print(f"Within ±0.5 log S:  {acc_05:.2f}%")
    print(f"Within ±1.0 log S:  {acc_10:.2f}%")
    print(f"Within ±1.5 log S:  {acc_15:.2f}%")
    print(f"Within ±2.0 log S:  {acc_20:.2f}%")

    final_results_df = pd.DataFrame([{
        "Model": f"GraphSAGE ({BEST_LAYERS}-layer)",
        "Test RMSE": test_rmse,
        "Test MAE": test_mae,
        "Test R2": test_r2,
        "Accuracy ±0.5 (%)": acc_05,
        "Accuracy ±1.0 (%)": acc_10,
        "Accuracy ±1.5 (%)": acc_15,
        "Accuracy ±2.0 (%)": acc_20,
    }])
    final_results_df.to_csv(results_dir / "final_model_results.csv", index=False)

    # Export training curves
    curves_df = pd.DataFrame({
        "Epoch": range(1, len(final_train_losses) + 1),
        "Train_MSE": final_train_losses,
        "Validation_MSE": final_val_losses,
        "Train_RMSE": np.sqrt(final_train_losses),
        "Validation_RMSE": np.sqrt(final_val_losses),
    })
    curves_df.to_csv(results_dir / "final_training_curves.csv", index=False)

    # Export molecule-level predictions & split IDs
    mol_predictions_df = pd.concat([
        get_split_predictions(df, final_model, train_graphs, X_train.index, "Train", device),
        get_split_predictions(df, final_model, val_graphs, X_val.index, "Validation", device),
        get_split_predictions(df, final_model, test_graphs, X_test.index, "Test", device),
    ], ignore_index=True)

    mol_predictions_df.to_csv(results_dir / "molecule_predictions.csv", index=False)
    mol_predictions_df[["Molecule_Index", "Compound ID", "smiles", "Split"]].to_csv(
        results_dir / "split_ids.csv", index=False
    )

    # 11. Split Investigation (Separating Split Effects from Model Changes)
    print("\n--- Step 9: Split Investigation ---")
    split_configs = [
        {"Split_Name": "70/15/15 (Seed 42)", "test_temp": 0.30, "seed": 42},
        {"Split_Name": "70/15/15 (Seed 100)", "test_temp": 0.30, "seed": 100},
        {"Split_Name": "80/10/10 (Seed 42)", "test_temp": 0.20, "seed": 42},
    ]

    split_comparison_rows = []
    for cfg in split_configs:
        s_name = cfg["Split_Name"]
        s_seed = cfg["seed"]
        t_size = cfg["test_temp"]

        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=t_size, random_state=s_seed)
        X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=s_seed)

        tr_graphs = create_graph_dataset(df, X_tr, y_tr)
        va_graphs = create_graph_dataset(df, X_va, y_va)
        te_graphs = create_graph_dataset(df, X_te, y_te)

        tr_loader = DataLoader(tr_graphs, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(va_graphs, batch_size=BATCH_SIZE, shuffle=False)
        te_loader = DataLoader(te_graphs, batch_size=BATCH_SIZE, shuffle=False)

        for model_label in ["GCN (Mean Pool)", "GraphSAGE (Sum Pool)"]:
            set_seed(SEED)
            if model_label == "GCN (Mean Pool)":
                m = GCN(hidden_channels=64, dropout=0.2).to(device)
            else:
                m = GraphSAGEFlexible(
                    num_layers=BEST_LAYERS, pooling="sum", hidden_channels=64, dropout=0.2
                ).to(device)

            fit_model(m, tr_loader, va_loader, LEARNING_RATE, EPOCHS, criterion, device)
            v_rmse, v_mae, _, _, _ = evaluate_metrics(m, va_loader, device)
            t_rmse, t_mae, t_r2, _, _ = evaluate_metrics(m, te_loader, device)

            split_comparison_rows.append({
                "Split Setting": s_name,
                "Model": model_label,
                "Train Size": len(tr_graphs),
                "Val Size": len(va_graphs),
                "Test Size": len(te_graphs),
                "Validation RMSE (log S)": v_rmse,
                "Validation MAE (log S)": v_mae,
                "Test RMSE (log S)": t_rmse,
                "Test MAE (log S)": t_mae,
                "Test R2": t_r2,
            })

    split_comp_df = pd.DataFrame(split_comparison_rows)
    split_comp_df.to_csv(results_dir / "split_comparison.csv", index=False)
    print(split_comp_df.to_string(index=False))
    print("\nAll training runs, checkpoints, and dashboard CSVs saved successfully!")


if __name__ == "__main__":
    main()
