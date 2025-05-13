import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit import RDLogger
from sklearn.metrics import r2_score
import joblib
import random
import datetime
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

# 설정
seed = 580
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 경로 설정
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
test_ids = test['ID'].copy()

# RDKit 기반 특징 생성 함수
def rdkit_features(df):
    phenol_group = Chem.MolFromSmarts("c[OH]")
    alcohol_group = Chem.MolFromSmarts("[CX4][OH]")
    ketone_group = Chem.MolFromSmarts("[CX3](=O)[#6]")
    thiol_ether = Chem.MolFromSmarts("[#6]-S-[#6]")
    aromatic_thiol_ether = Chem.MolFromSmarts("c-S-[#6]")
    sulfonyl_group = Chem.MolFromSmarts("S(=O)(=O)")
    phosphates_group = Chem.MolFromSmiles("P(=O)")
    amide_group = Chem.MolFromSmarts("[NX3][CX3](=O)[#6]")

    mols = df['Canonical_Smiles'].apply(Chem.MolFromSmiles)
    df = df.copy()
    df['Mol'] = mols
    df['MolLogP'] = mols.apply(Crippen.MolLogP)
    df['NumHAcceptors'] = mols.apply(Lipinski.NumHAcceptors)
    df['NumHDonors'] = mols.apply(Lipinski.NumHDonors)
    df['TPSA'] = mols.apply(Descriptors.TPSA)
    df['NumAromaticRings'] = mols.apply(Lipinski.NumAromaticRings)
    df['NumAromaticHeterocycles'] = mols.apply(Lipinski.NumAromaticHeterocycles)
    df['NumSaturatedHeterocycles'] = mols.apply(Lipinski.NumSaturatedHeterocycles)
    df['NOCount'] = mols.apply(Descriptors.NOCount)
    df['NumRadicalElectrons'] = mols.apply(Descriptors.NumRadicalElectrons)
    df['NHOHCount'] = mols.apply(Descriptors.NHOHCount)
    df['HasPhenol'] = mols.apply(lambda m: m.HasSubstructMatch(phenol_group)).astype(int)
    df['HasAlcohol'] = mols.apply(lambda m: m.HasSubstructMatch(alcohol_group)).astype(int)
    df['HasKetone'] = mols.apply(lambda m: m.HasSubstructMatch(ketone_group)).astype(int)
    df['HasThioether'] = mols.apply(lambda m: m.HasSubstructMatch(thiol_ether)).astype(int)
    df['HasArylThioether'] = mols.apply(lambda m: m.HasSubstructMatch(aromatic_thiol_ether)).astype(int)
    df['HasSulfonyl'] = mols.apply(lambda m: m.HasSubstructMatch(sulfonyl_group)).astype(int)
    df['HasPhosphate'] = mols.apply(lambda m: m.HasSubstructMatch(phosphates_group)).astype(int)
    df['HasAmide'] = mols.apply(lambda m: m.HasSubstructMatch(amide_group)).astype(int)
    return df

train = rdkit_features(train)
test = rdkit_features(test)

atom_features = {
    'H': [1.0, 0, 0, 0],
    'C': [0, 1.0, 0, 0],
    'N': [0, 0, 1.0, 0],
    'O': [0, 0, 0, 1.0]
}

def mol_to_graph_data_obj(mol, y=None):
    node_feats = []
    edge_index = [[], []]

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        node_feats.append(atom_features.get(symbol, [0, 0, 0, 0]))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]

    data = Data(x=torch.tensor(node_feats, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor([y], dtype=torch.float) if y is not None else None)
    return data

train_graphs = [mol_to_graph_data_obj(m, y) for m, y in zip(train['Mol'], train['Inhibition']) if m is not None]
test_graphs = [mol_to_graph_data_obj(m) for m in test['Mol'] if m is not None]

# GNN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x.view(-1)

# 학습 준비
batch_size = 32
epochs = 2000
patience = 100
best_val_loss = float('inf')
patience_counter = 0
loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

model = GCN(input_dim=4, hidden_dim=64, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = torch.nn.MSELoss()

# 학습 루프
model.train()
train_losses = []
best_model_state = None
for epoch in range(epochs):
    total_loss = 0
    all_preds = []
    all_targets = []
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(out.detach().cpu().numpy())
        all_targets.extend(batch.y.view(-1).detach().cpu().numpy())

    r2 = r2_score(all_targets, all_preds)
    train_losses.append(total_loss)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, R2: {r2:.4f}")

    if total_loss < best_val_loss:
        best_val_loss = total_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# 최적 모델 로드
if best_model_state:
    model.load_state_dict(best_model_state)

# 예측
model.eval()
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
preds = []
with torch.no_grad():
    for batch in test_loader:
        pred = model(batch)
        preds += pred.tolist()

submission = pd.DataFrame({
    'ID': test_ids[:len(preds)],
    'Inhibition': preds
})

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = path + f'gnn_model_{now}.pt'
submission_path = path + f'submission_gnn_{now}.csv'

torch.save(model.state_dict(), model_path)
submission.to_csv(submission_path, index=False)

print(f"\n✔️ GNN 모델 저장 완료: {model_path}")
print(f"📄 제출 파일 저장 완료: {submission_path}")

# Epoch 140/200, Loss: 33464.7794, R2: 0.0965
# Early stopping triggered.
# ✔️ GNN 모델 저장 완료: C:/Study25/_data/dacon/boost_up/gnn_model_20250717_203808.pt
# 📄 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_gnn_20250717_203808.csv


# Epoch 930/2000, Loss: 32701.8711, R2: 0.1144
# Early stopping triggered.
# ✔️ GNN 모델 저장 완료: C:/Study25/_data/dacon/boost_up/gnn_model_20250717_204548.pt
# 📄 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_gnn_20250717_204548.csv

# Epoch 1400/2000, Loss: 31218.6117, R2: 0.1534
# Early stopping triggered.
# ✔️ GNN 모델 저장 완료: C:/Study25/_data/dacon/boost_up/gnn_model_20250717_212430.pt
# 📄 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_gnn_20250717_212430.csv