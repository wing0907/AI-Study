import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdPartialCharges, rdMolDescriptors, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect  # Updated to avoid deprecation warning
from rdkit import RDLogger
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import random
import datetime
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

def define_substructures():
    return {
        'HasPhenol': Chem.MolFromSmarts("c[OH]"),
        'HasAlcohol': Chem.MolFromSmarts("[CX4][OH]"),
        'HasKetone': Chem.MolFromSmarts("[CX3](=O)[#6]"),
        'HasThioether': Chem.MolFromSmarts("[#6]-S-[#6]"),
        'HasArylThioether': Chem.MolFromSmarts("c-S-[#6]"),
        'HasSulfonyl': Chem.MolFromSmarts("S(=O)(=O)"),
        'HasPhosphate': Chem.MolFromSmiles("P(=O)"),
        'HasAmide': Chem.MolFromSmarts("[NX3][CX3](=O)[#6]")
    }

substructure_smarts = define_substructures()


def get_physchem_features(mol):
    return [
        Descriptors.MolWt(mol), Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol),
        Descriptors.HeavyAtomMolWt(mol), Descriptors.MolMR(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol), Descriptors.NOCount(mol), Descriptors.NHOHCount(mol),
        Descriptors.RingCount(mol), Descriptors.NumAliphaticRings(mol), Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol), Descriptors.NumValenceElectrons(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.NumRadicalElectrons(mol), Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumAliphaticHeterocycles(mol), Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.TPSA(mol), rdMolDescriptors.CalcLabuteASA(mol), Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol), Descriptors.FpDensityMorgan3(mol), rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.BertzCT(mol), Descriptors.HallKierAlpha(mol), Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHeteroatoms(mol), Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol),
        Descriptors.MaxAbsPartialCharge(mol), Descriptors.MinAbsPartialCharge(mol), Descriptors.Ipc(mol)
    ]


def get_morgan_fp(mol, radius=2, n_bits=512):
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)  # Updated function call
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        physchem = get_physchem_features(mol)
        morgan = get_morgan_fp(mol)
        substructure_flags = [int(mol.HasSubstructMatch(smarts)) for smarts in substructure_smarts.values()]
        return physchem + list(morgan) + substructure_flags
    except:
        return None


def extract_features_parallel(df, num_workers=4):
    pool = multiprocessing.Pool(processes=num_workers)
    features = pool.map(smiles_to_features, df['Canonical_Smiles'])
    pool.close()
    pool.join()
    return features


def main():
    RDLogger.DisableLog('rdApp.*')
    seed = 592
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    path = 'C:/Study25/_data/dacon/boost_up/'
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")
    test_ids = test['ID'].copy()

    train['Inhibition_log'] = np.log1p(train['Inhibition'])
    y_mean = train['Inhibition_log'].mean()
    y_std = train['Inhibition_log'].std()
    train['Inhibition_scaled'] = (train['Inhibition_log'] - y_mean) / y_std

    train_feats = extract_features_parallel(train)
    test_feats = extract_features_parallel(test)

    scaler = StandardScaler()
    train_feats = np.array([f for f in train_feats if f is not None])
    test_feats = np.array([f for f in test_feats if f is not None])
    train_scaled = scaler.fit_transform(train_feats)
    test_scaled = scaler.transform(test_feats)

    mol_list_train = [Chem.MolFromSmiles(s) for s in train['Canonical_Smiles']]
    mol_list_test = [Chem.MolFromSmiles(s) for s in test['Canonical_Smiles']]

    atom_features = {'H': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'N': [0, 0, 1, 0], 'O': [0, 0, 0, 1]}

    def mol_to_graph(mol, feat_row, y=None):
        node_feats = [atom_features.get(a.GetSymbol(), [0, 0, 0, 0]) for a in mol.GetAtoms()]
        edge_index = [[], []]
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0] += [i, j]
            edge_index[1] += [j, i]
        node_feats = [nf + list(feat_row) for nf in node_feats]
        return Data(x=torch.tensor(node_feats, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor([y], dtype=torch.float) if y is not None else None)

    X_graphs = [mol_to_graph(m, f, y) for m, f, y in zip(mol_list_train, train_scaled, train['Inhibition_scaled']) if m is not None]
    T_graphs = [mol_to_graph(m, f) for m, f in zip(mol_list_test, test_scaled) if m is not None]

    class GCN(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.fc = torch.nn.Linear(hidden_dim, 1)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.fc(x).view(-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    all_preds, all_targets, all_test_preds = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_graphs)):
        model = GCN(X_graphs[0].x.size(1), 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        train_loader = DataLoader([X_graphs[i] for i in tr_idx], batch_size=3, shuffle=True)
        val_loader = DataLoader([X_graphs[i] for i in val_idx], batch_size=3)

        best_loss, best_model, patience, counter = float('inf'), None, 100, 0
        for epoch in range(2000):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = loss_fn(out, batch.y.view(-1))
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss, val_preds, val_true = 0, [], []
            for batch in val_loader:
                with torch.no_grad():
                    out = model(batch)
                    val_loss += loss_fn(out, batch.y.view(-1)).item()
                    val_preds += out.tolist()
                    val_true += batch.y.view(-1).tolist()
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        model.load_state_dict(best_model)
        model.eval()
        val_preds_rescaled = np.expm1(np.array(val_preds) * y_std + y_mean)
        val_true_rescaled = np.expm1(np.array(val_true) * y_std + y_mean)
        val_preds_rescaled = np.clip(val_preds_rescaled, 0, 100)
        all_preds += list(val_preds_rescaled)
        all_targets += list(val_true_rescaled)

        test_loader = DataLoader(T_graphs, batch_size=3)
        fold_preds = []
        for batch in test_loader:
            with torch.no_grad():
                out = model(batch)
                fold_preds += out.tolist()
        fold_preds_rescaled = np.expm1(np.array(fold_preds) * y_std + y_mean)
        fold_preds_rescaled = np.clip(fold_preds_rescaled, 0, 100)
        all_test_preds.append(fold_preds_rescaled)

    final_test_preds = np.mean(all_test_preds, axis=0)

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = path + f'gnn_kfold_ensemble_{now}.pt'
    submission_path = path + f'submission_gnn_kfold_{now}.csv'

    submission = pd.DataFrame({
        'ID': test_ids[:len(final_test_preds)],
        'Inhibition': final_test_preds
    })
    submission.to_csv(submission_path, index=False)
    torch.save(model.state_dict(), model_path)

    print(f"\n✔️ GNN KFold 앙상블 모델 저장: {model_path}")
    print(f"📄 제출 파일 저장: {submission_path}")
    print(f"📈 전체 R2 Score: {r2_score(all_targets, all_preds):.4f}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


# ✔️ GNN KFold 앙상블 모델 저장: C:/Study25/_data/dacon/boost_up/gnn_kfold_ensemble_20250722_101057.pt
# 📄 제출 파일 저장: C:/Study25/_data/dacon/boost_up/submission_gnn_kfold_20250722_101057.csv
# 📈 전체 R2 Score: -0.1956

# ✔️ GNN KFold 앙상블 모델 저장: C:/Study25/_data/dacon/boost_up/gnn_kfold_ensemble_20250722_102001.pt
# 📄 제출 파일 저장: C:/Study25/_data/dacon/boost_up/submission_gnn_kfold_20250722_102001.csv
# 📈 전체 R2 Score: -0.1956

# ✔️ GNN KFold 앙상블 모델 저장: C:/Study25/_data/dacon/boost_up/gnn_kfold_ensemble_20250722_112950.pt
# 📄 제출 파일 저장: C:/Study25/_data/dacon/boost_up/submission_gnn_kfold_20250722_112950.csv
# 📈 전체 R2 Score: -0.1842