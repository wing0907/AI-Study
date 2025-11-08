import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdPartialCharges, rdMolDescriptors
from rdkit import RDLogger
from sklearn.metrics import r2_score
import random
import datetime
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog('rdApp.*')

# 설정
seed = 592
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 경로 설정
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
test_ids = test['ID'].copy()

# 타겟 로그 변환
train['Inhibition_log'] = np.log1p(train['Inhibition'])
y_mean = train['Inhibition_log'].mean()
y_std = train['Inhibition_log'].std()
train['Inhibition_scaled'] = (train['Inhibition_log'] - y_mean) / y_std

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
    df['FractionCSP3'] = mols.apply(Descriptors.FractionCSP3)

    def calc_partial_charges(m):
        try:
            rdPartialCharges.ComputeGasteigerCharges(m)
            charges = [float(a.GetProp('_GasteigerCharge')) for a in m.GetAtoms() if a.HasProp('_GasteigerCharge')]
            if len(charges) == 0:
                return [0, 0, 0, 0]
            max_charge = max(charges)
            min_charge = min(charges)
            max_abs = max(abs(c) for c in charges)
            min_abs = min(abs(c) for c in charges)
            return [max_charge, min_charge, max_abs, min_abs]
        except:
            return [0, 0, 0, 0]

    partial_charge_features = mols.apply(calc_partial_charges)
    df['MaxPartialCharge'] = partial_charge_features.apply(lambda x: x[0])
    df['MinPartialCharge'] = partial_charge_features.apply(lambda x: x[1])
    df['MaxAbsPartialCharge'] = partial_charge_features.apply(lambda x: x[2])
    df['MinAbsPartialCharge'] = partial_charge_features.apply(lambda x: x[3])

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

rdkit_feature_cols = [
    'MolLogP', 'NumHAcceptors', 'NumHDonors', 'TPSA', 'NumAromaticRings',
    'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NOCount',
    'NumRadicalElectrons', 'NHOHCount', 'FractionCSP3',
    'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'HasPhenol', 'HasAlcohol', 'HasKetone', 'HasThioether', 'HasArylThioether',
    'HasSulfonyl', 'HasPhosphate', 'HasAmide'
]

scaler = StandardScaler()
train_rdkit_feats = scaler.fit_transform(train[rdkit_feature_cols])
test_rdkit_feats = scaler.transform(test[rdkit_feature_cols])

atom_features = {
    'H': [1.0, 0, 0, 0],
    'C': [0, 1.0, 0, 0],
    'N': [0, 0, 1.0, 0],
    'O': [0, 0, 0, 1.0]
}

def mol_to_graph_data_obj(mol, rdkit_feat_row, y=None):
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

    node_feats = [nf + list(rdkit_feat_row) for nf in node_feats]

    data = Data(x=torch.tensor(node_feats, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor([y], dtype=torch.float) if y is not None else None)
    return data

train_graphs = [mol_to_graph_data_obj(m, r, y) for m, r, y in zip(train['Mol'], train_rdkit_feats, train['Inhibition_scaled']) if m is not None]
test_graphs = [mol_to_graph_data_obj(m, r) for m, r in zip(test['Mol'], test_rdkit_feats) if m is not None]

# 모델 정의
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

# 학습
model = GCN(input_dim=train_graphs[0].x.size(1), hidden_dim=64, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
loader = DataLoader(train_graphs, batch_size=3, shuffle=True)

model.train()
best_loss = float('inf')
best_state = None
patience = 200
counter = 0
all_preds_final = []
all_targets_final = []

for epoch in range(2000):
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
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, R2: {r2:.4f}")

    if total_loss < best_loss:
        best_loss = total_loss
        best_state = model.state_dict()
        all_preds_final = all_preds
        all_targets_final = all_targets
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# 모델 저장 및 예측
if best_state:
    model.load_state_dict(best_state)

print(f"\n✔️ 최종 GNN R2 Score: {r2_score(all_targets_final, all_preds_final):.4f}")

model.eval()
test_loader = DataLoader(test_graphs, batch_size=3, shuffle=False)
preds = []
with torch.no_grad():
    for batch in test_loader:
        pred = model(batch)
        preds += pred.tolist()

# 로그 역변환
preds = np.expm1(np.array(preds) * y_std + y_mean)
preds = np.clip(preds, 0, 100)

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = path + f'gnn_model_{now}.pt'
submission_path = path + f'submission_gnn_{now}.csv'

torch.save(model.state_dict(), model_path)
submission = pd.DataFrame({
    'ID': test_ids[:len(preds)],
    'Inhibition': preds
})
submission.to_csv(submission_path, index=False)

print(f"\n📦 모델 저장 완료: {model_path}")
print(f"📄 제출 파일 저장 완료: {submission_path}")


# ✔️ 최종 GNN R2 Score: 0.9954
# 📦 모델 저장 완료: C:/Study25/_data/dacon/boost_up/gnn_model_20250721_173349.pt
# 📄 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_gnn_20250721_173349.csv



# ✔️ 최종 GNN R2 Score: 0.9912
# 📦 모델 저장 완료: C:/Study25/_data/dacon/boost_up/gnn_model_20250721_182301.pt
# 📄 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_gnn_20250721_182301.csv