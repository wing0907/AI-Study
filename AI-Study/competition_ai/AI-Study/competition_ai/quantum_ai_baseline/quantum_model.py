# quantum_model.py
import pennylane as qml
import torch
from pennylane.qnn import TorchLayer

# 1) Device 설정: default.qubit + backprop
#    Autograd 미분 사용으로 배치 파라미터 그라디언트 오류 방지
dev = qml.device("default.qubit", wires=4)

# 2) QNode 정의
@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    """
    inputs: tensor of shape (batch_size, 4)
    weights: tensor of shape (2, 4, 3)
    returns: list of 4 expectation values
    """
    # Angle Embedding: 4차원 입력 → 4 qubit rotations
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    # Variational layers: 2 layers, strongly entangling
    qml.templates.StronglyEntanglingLayers(weights, wires=range(4))
    # Measure PauliZ expectation on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# 3) 파라미터 형태 정의: 2 layers × 4 qubits × 3 params each
weight_shapes = {"weights": (2, 4, 3)}  

# 4) 리소스 제약 검증
#    Dummy inputs/weights로 스펙 추출 후 assertion
dummy_inputs  = torch.zeros(4, dtype=torch.float64)
dummy_weights = torch.zeros(2, 4, 3, dtype=torch.float64)
specs = qml.specs(circuit)(dummy_inputs, dummy_weights)
assert specs["num_tape_wires"] <= 8,        f"큐빗 수 초과: {specs['num_tape_wires']} > 8"
assert specs['resources'].depth <= 30,       f"회로 깊이 초과: {specs['resources'].depth} > 30"
# assert specs['resources'].depth <= 30, f"회로 깊이 초과: {specs['resources'].depth} > 30"

assert specs["num_trainable_params"] <= 60, f"퀀텀 파라미터 수 초과: {specs['num_trainable_params']} > 60"

# 5) PyTorch용 QNN 레이어 생성
qnn_torch = TorchLayer(circuit, weight_shapes)