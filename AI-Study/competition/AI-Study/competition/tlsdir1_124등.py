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
# 0. Seed 리스트 (단일 seed)
# -----------------------------
SEEDS = [162]

# -----------------------------
# 1. 데이터 로드
# -----------------------------
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# -----------------------------
# 2. 타겟 로그 변환  # 억제율(Inhibition)이 양수 skewed 분포 → log1p로 정규화
# -----------------------------
train['Inhibition_log'] = np.log1p(train['Inhibition']) # np.log1p(x) = log(1 + x), 분포를 압축해서 (스케일 줄이기 + 왜도 감소)
y_mean = train['Inhibition_log'].mean() # 로그변환된 억제값의 평균
y_std = train['Inhibition_log'].std() # 로그변환된 억제값의 표준편차
train['Inhibition_scaled'] = (train['Inhibition_log'] - y_mean) / y_std # 스케일링 = 각 값을 평균에서 빼고 표준편차로 나누면 평균=0, 표준편차=1인 분포로 변환됨

# -----------------------------
# 3. SMILES → Graph 변환 함수 # [원자번호, 결합 차수, 혼성화 상태, 방향족 여부, 전하, 질량, 반데르발스 반지름]
# -----------------------------
def atom_features(atom):
    """노드 피처: 원자번호, 결합 차수, 혼성화, 아로마틱, 공식 전하, 질량, VDW 반지름"""
    return [
        atom.GetAtomicNum(), #원자 종류를 알려주는 가장 중요한 값
        atom.GetDegree(), #이 원자가 다른 원자 몇 개랑 연결돼 있는지
        atom.GetHybridization().real, #sp, sp2, sp3 같은 원자 궤도 혼성화 / 결합 각도와 결합 성질에 큰 영향을 줌
        atom.GetIsAromatic(), # 방향족 여부 / 방향족성은 분자의 화학적 성질과 안정성에 크게 기여햄
        atom.GetFormalCharge(), #원자의 정식 전하 / 이온화 상태를 구분하는 데 중요 ㅇㅇ
        atom.GetMass(), #원자 질량 / 질량 정보는 전체 분자 질량 계산에 사용
        rdchem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) #반데르발스 반지름 / 분자 간 상호작용과 공간적 충돌을 판단할 때 중요!
    ]
    
    # 이 함수는 원자 하나를 “수치 벡터”로 바꿔서 GNN이 처리할 수 있게 하는 변환 함수
    # 모델이 SMILES 문자를 이해하게 만드는 가장 첫 번째 전처리 단계라고 늬낌
    

def bond_features(bond): #bond_features()는 결합의 화학적 성질을 [3개 숫자] 벡터로 바꾸는 함수
    """엣지 피처: 결합 타입, IsInRing, IsConjugated"""
    bt = bond.GetBondType() 
    return [
        {
            Chem.rdchem.BondType.SINGLE: 0,  #SINGLE: 단일결합 → 0
            Chem.rdchem.BondType.DOUBLE: 1,  #DOUBLE: 이중결합 → 1
            Chem.rdchem.BondType.TRIPLE: 2,  #TRIPLE: 삼중결합 → 2
            Chem.rdchem.BondType.AROMATIC: 3 #AROMATIC: 방향족 결합 → 3  ex)C=C 결합이면 DOUBLE → 1 / C-C 결합이면 SINGLE → 0 (핵심결합정보)
        }.get(bt, -1),
        bond.IsInRing(), #이 결합이 고리(ring) 안에 있?? -> 고리 속하면 트루, 안 속하면 펄스
        bond.GetIsConjugated() #결합이 공명(conjugation) 상태인?? -> 예: C=O 결합 옆에 있는 C-C 결합 (Ture, False)
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
print("Test graphs:", len(test_graphs))

# -----------------------------
# 5. MPNN 모델 정의
# -----------------------------
class MPNN(torch.nn.Module):
    def __init__(self, in_channels, edge_channels):
        super(MPNN, self).__init__()
        nn1 = Sequential(Linear(edge_channels, 32), ReLU(), Linear(32, in_channels * 64))
        self.conv1 = NNConv(in_channels, 64, nn1, aggr='mean')
        self.bn1 = BatchNorm1d(64)
        nn2 = Sequential(Linear(edge_channels, 32), ReLU(), Linear(32, 64 * 128))
        self.conv2 = NNConv(64, 128, nn2, aggr='mean')
        self.bn2 = BatchNorm1d(128)
        self.lin1 = Linear(128, 64)
        self.lin2 = Linear(64, 1)
        self.dropout = Dropout(0.5) 

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
print("Using device:", device)

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
        print(f"\n--- Fold {fold+1} ---")

        tr_df = train.iloc[tr_idx]
        aug_graphs = []
        for s, y in zip(tr_df['Canonical_Smiles'], tr_df['Inhibition_scaled']):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            g = smiles_to_data(s, y)
            if g:
                aug_graphs.append(g)
            for _ in range(10):
                r_smi = Chem.MolToSmiles(mol, doRandom=True)
                g_aug = smiles_to_data(r_smi, y)
                if g_aug:
                    aug_graphs.append(g_aug)

        va_graphs = [smiles_to_data(s, y) for s, y in zip(train.iloc[va_idx]['Canonical_Smiles'], train.iloc[va_idx]['Inhibition_scaled'])]
        va_graphs = [g for g in va_graphs if g is not None]

        model = MPNN(in_channels=7, edge_channels=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = MSELoss()

        tr_loader = DataLoader(aug_graphs, batch_size=32, shuffle=True)
        va_loader = DataLoader(va_graphs, batch_size=32)

        best_rmse = np.inf
        patience, patience_limit = 0, 15

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
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), f"{path}mpnn_fold{fold+1}.pt")
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    print("Early stopping.")
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

# -----------------------------
# 8. 제출 파일
# -----------------------------
test_preds_rescaled = np.expm1(test_preds_ensemble * y_std + y_mean)
submission = pd.DataFrame({'ID': test['ID'], 'Inhibition': test_preds_rescaled})
submission.to_csv(path + 'submission_0703_1020.csv', index=False)
print("submission_mpnn_clean.csv 저장 완료.")


# === 최종 앙상블 OOF 성능 ===
# RMSE: 29.2595
# R2: 0.1286
# MAE: 22.5981
# submission_mpnn_clean.csv 저장 완료.



# === 최종 앙상블 OOF 성능 ===
# RMSE: 29.2595
# R2: -0.2286
# MAE: 22.5981
# submission_mpnn_clean.csv 저장 완료.


# === 최종 앙상블 OOF 성능 ===
# RMSE: 28.2460
# R2: -0.1449
# MAE: 21.8494
# submission_mpnn_clean.csv 저장 완료.