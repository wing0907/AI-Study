import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdchem
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random

import torch
from torch.nn import Linear, MSELoss, Dropout, BatchNorm1d, ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data

# -----------------------------
# 0. Seed 리스트
# -----------------------------
SEEDS = [592]

# -----------------------------
# 1. 데이터 로드
# -----------------------------
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# -----------------------------
# 2. 타겟 로그 변환
# -----------------------------
train['Inhibition_log'] = np.log1p(train['Inhibition'])
y_mean = train['Inhibition_log'].mean()
y_std = train['Inhibition_log'].std()
train['Inhibition_scaled'] = (train['Inhibition_log'] - y_mean) / y_std

# -----------------------------
# 3. SMILES → Graph 변환
# -----------------------------
def atom_features(atom):
    max_atomic_num = 20
    onehot = [0]*max_atomic_num
    atomic_num = atom.GetAtomicNum()
    if atomic_num <= max_atomic_num:
        onehot[atomic_num - 1] = 1
    else:
        onehot[-1] = 1
    features = onehot + [
        atom.GetDegree(),
        atom.GetHybridization().real,
        atom.GetIsAromatic(),
        atom.GetFormalCharge(),
        atom.GetMass(),
        rdchem.GetPeriodicTable().GetRvdw(atomic_num)
    ]
    return features


def bond_features(bond):
    bt = bond.GetBondType()
    return [
        {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3
        }.get(bt, -1),
        bond.IsInRing(),
        bond.GetIsConjugated()
    ]

def smiles_to_data(smiles, y=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_idx, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_idx += [[i, j], [j, i]]
        edge_feats += [feat, feat]
    if len(edge_idx) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=None if y is None else torch.tensor([y], dtype=torch.float)
    )
    return data

# -----------------------------
# 4. 테스트셋 그래프 생성
# -----------------------------
test_graphs = [smiles_to_data(s) for s in test['Canonical_Smiles']]
test_graphs = [g for g in test_graphs if g is not None]

# -----------------------------
# 5. MPNN 모델 정의 (튜닝 파라미터 적용)
# -----------------------------
class MPNN(torch.nn.Module):
    def __init__(self, in_channels, edge_channels):
        super(MPNN, self).__init__()
        nn1 = Sequential(
            Linear(edge_channels, 32),
            ReLU(),
            Linear(32, in_channels * 120)   # hidden_dim1=120
        )
        self.conv1 = NNConv(in_channels, 120, nn1, aggr='mean')
        self.bn1 = BatchNorm1d(120)
        nn2 = Sequential(
            Linear(edge_channels, 32),
            ReLU(),
            Linear(32, 120 * 210)           # hidden_dim2=210
        )
        self.conv2 = NNConv(120, 210, nn2, aggr='mean')
        self.bn2 = BatchNorm1d(210)
        self.lin1 = Linear(210, 64)
        self.lin2 = Linear(64, 1)
        self.dropout = Dropout(0.3250)     # dropout=0.325

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(F.relu(self.lin1(x)))
        x = self.lin2(x)
        return x.squeeze(1)

# -----------------------------
# 6. 학습 및 예측
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_preds_ensemble = np.zeros(len(test_graphs))
oof_ensemble = np.zeros(len(train))

for SEED in SEEDS:
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test_graphs))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
        tr_df = train.iloc[tr_idx]
        aug_graphs = []
        for s, y in zip(tr_df['Canonical_Smiles'], tr_df['Inhibition_scaled']):
            g = smiles_to_data(s, y)
            if g:
                aug_graphs.append(g)
            for _ in range(10):
                r_smi = Chem.MolToSmiles(Chem.MolFromSmiles(s), doRandom=True)
                g_aug = smiles_to_data(r_smi, y)
                if g_aug:
                    aug_graphs.append(g_aug)

        va_graphs = [smiles_to_data(s, y) for s, y in zip(train.iloc[va_idx]['Canonical_Smiles'], train.iloc[va_idx]['Inhibition_scaled'])]
        va_graphs = [g for g in va_graphs if g is not None]
        
        
        model = MPNN(
            in_channels=len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0))),
            edge_channels=3
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00726)  # learning_rate=0.00726
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = MSELoss()

        tr_loader = DataLoader(aug_graphs, batch_size=32, shuffle=True)
        va_loader = DataLoader(va_graphs, batch_size=32)

        best_rmse = np.inf
        patience = 0
        patience_limit = 15

        for epoch in range(1, 101):
            model.train()
            total_loss = 0
            for batch in tr_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            train_loss = total_loss / len(tr_loader.dataset)

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in va_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    preds.extend(out.cpu().numpy())
                    targets.extend(batch.y.cpu().numpy())
            val_rmse = np.sqrt(mean_squared_error(targets, preds))
            scheduler.step(val_rmse)

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), f"{path}mpnn_fold{fold+1}.pt")
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    break

        model.load_state_dict(torch.load(f"{path}mpnn_fold{fold+1}.pt"))
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                preds.extend(out.cpu().numpy())
        oof[va_idx] = preds

        test_loader = DataLoader(test_graphs, batch_size=32)
        fold_test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                fold_test_preds.extend(out.cpu().numpy())
        test_preds += np.array(fold_test_preds) / kf.n_splits

    oof_ensemble += oof / len(SEEDS)
    test_preds_ensemble += test_preds / len(SEEDS)

# -----------------------------
# 7. 최종 성능 (로그 역변환)
# -----------------------------
oof_rescaled = np.expm1(oof_ensemble * y_std + y_mean)
rmse = np.sqrt(mean_squared_error(train['Inhibition'], oof_rescaled))
r2 = r2_score(train['Inhibition'], oof_rescaled)
mae = mean_absolute_error(train['Inhibition'], oof_rescaled)
print("\n=== 최종 앙상블 OOF 성능 ===")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")

"""
점수계산용
print(f"\nMax Inhibition: {train['Inhibition'].max():.4f}")
print(f"Min Inhibition: {train['Inhibition'].min():.4f}")
correlation = np.corrcoef(train['Inhibition'], oof_rescaled)[0, 1]
print(f"선형 상관계수: {correlation:.4f}")

Max Inhibition: 99.3815
Min Inhibition: 0.0000
선형 상관계수: 0.1264

A
range_inhibition = 99.3815 - 0.0 = 99.3815
normalized_rmse = rmse / range_inhibition = 27.4182 / 99.3815 ≈ 0.2759

B = 0.1264  # corr 값 그대로

Score = 0.5 * (1 - min(0.2759, 1)) + 0.5 * 0.1264
       = 0.5 * 0.7241 + 0.5 * 0.1264
       ≈ 0.36205 + 0.0632
       = 0.4253

Score ≈ 0.4253
"""
# -----------------------------
# 8. 제출 파일
# -----------------------------
test_preds_rescaled = np.expm1(test_preds_ensemble * y_std + y_mean)
submission = pd.DataFrame({'ID': test['ID'], 'Inhibition': test_preds_rescaled})
submission.to_csv(path + 'submission_4.csv', index=False)


#17 seed 592
# RMSE: 27.4182
# R2: -0.0788
# MAE: 21.3757
