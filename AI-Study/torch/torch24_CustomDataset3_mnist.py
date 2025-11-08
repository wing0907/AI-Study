import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as tr

# --------------------------
# 1. 변환 (transform)
# --------------------------
transf = tr.Compose([
    tr.Resize(56),                # 28x28 → 56x56
    tr.ToTensor(),                # [0,255] → [0,1]
    tr.Normalize((0.5,), (0.5,))  # 평균 0.5, 표준편차 0.5로 정규화
])

# --------------------------
# 2. MNIST 데이터셋 다운로드 & 로드
# --------------------------
path = './_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)

# --------------------------
# 3. DataLoader
# --------------------------
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --------------------------
# 4. 데이터 확인
# --------------------------
for batch_idx, (xb, yb) in enumerate(train_loader):
    print(f"배치 {batch_idx} → 이미지 shape: {xb.shape}, 라벨: {yb[:10]}")
    if batch_idx == 0:  # 첫 배치만 확인
        break
    
# 배치 0 → 이미지 shape: torch.Size([64, 1, 56, 56]), 라벨: tensor([9, 1, 1, 2, 6, 7, 5, 5, 9, 7])