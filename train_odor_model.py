"""Train a multi-label odor predictor from Morgan fingerprints."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# Dùng backend không cần cửa sổ để luôn xuất được ảnh khi chạy từ Terminal.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_SIZE = 2048
OUTPUT_SIZE = 113
HIDDEN_SIZE_1 = 1024
HIDDEN_SIZE_2 = 512
DROPOUT = 0.3

TRAIN_RATIO = 0.8
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42

DATASET_PATH = Path("odor_morgan_tensor_dataset.pt")
SMILES_CSV_PATH = Path("clean_dataset.csv")
BEST_WEIGHTS_PATH = Path("odor_predictor_weights.pth")
LEARNING_CURVE_PATH = Path("learning_curve.png")


class OdorPredictor(nn.Module):
    """Fully connected multi-label classifier for Morgan fingerprints."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Return raw logits. BCEWithLogitsLoss applies sigmoid internally.
        return self.network(features)


def choose_device() -> torch.device:
    """Prefer Apple MPS and fall back to CPU; CUDA is intentionally unused."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def validate_loaded_data(
    features: torch.Tensor,
    targets: torch.Tensor,
    label_names: list[str],
) -> None:
    """Fail early when the serialized dataset does not match the model contract."""
    if features.ndim != 2 or features.shape[1] != INPUT_SIZE:
        raise ValueError(
            f"X phải có shape [N, {INPUT_SIZE}], nhưng nhận được {tuple(features.shape)}"
        )
    if targets.ndim != 2 or targets.shape[1] != OUTPUT_SIZE:
        raise ValueError(
            f"Y phải có shape [N, {OUTPUT_SIZE}], nhưng nhận được {tuple(targets.shape)}"
        )
    if features.shape[0] != targets.shape[0]:
        raise ValueError("X và Y phải có cùng số lượng mẫu.")
    if len(label_names) != OUTPUT_SIZE:
        raise ValueError(
            f"label_names phải có {OUTPUT_SIZE} nhãn, nhưng nhận được {len(label_names)}"
        )


def smiles_to_chiral_morgan(smiles: str) -> torch.Tensor:
    """Chuyển một SMILES thành Morgan radius=2, 2048 bit có chirality."""
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"SMILES không hợp lệ: {smiles!r}")

    # Giữ đúng cấu hình inference của app.py để train và dự đoán đồng nhất.
    with rdBase.BlockLogs():
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius=2,
            nBits=INPUT_SIZE,
            useChirality=True,
        )
    fingerprint_array = np.zeros((INPUT_SIZE,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    return torch.from_numpy(fingerprint_array)


def create_chiral_features_from_csv(
    csv_path: Path,
    expected_rows: int,
) -> torch.Tensor:
    """Tái tạo X từ SMILES để không dùng lại fingerprint cũ thiếu chirality."""
    dataframe = pd.read_csv(csv_path)
    smiles_column = next(
        (column for column in dataframe.columns if column.lower() == "smiles"),
        None,
    )
    if smiles_column is None:
        raise ValueError(f"{csv_path} không có cột SMILES.")
    if len(dataframe) != expected_rows:
        raise ValueError(
            "Số hàng SMILES phải khớp với Y trong TensorDataset: "
            f"CSV={len(dataframe)}, Y={expected_rows}."
        )

    fingerprints: list[torch.Tensor] = []
    for row_number, value in enumerate(dataframe[smiles_column], start=2):
        if pd.isna(value):
            raise ValueError(f"SMILES bị thiếu tại dòng CSV {row_number}.")
        try:
            fingerprints.append(smiles_to_chiral_morgan(str(value).strip()))
        except ValueError as error:
            raise ValueError(f"Lỗi tại dòng CSV {row_number}: {error}") from error

    return torch.stack(fingerprints).float()


def plot_learning_curve(
    train_losses: list[float],
    test_losses: list[float],
    output_path: Path,
) -> None:
    """Vẽ và lưu Learning Curve của toàn bộ epoch thực sự đã chạy."""
    if not train_losses or len(train_losses) != len(test_losses):
        raise ValueError("Lịch sử Train/Test Loss không hợp lệ để vẽ biểu đồ.")

    epochs = list(range(1, len(train_losses) + 1))
    best_epoch = min(range(len(test_losses)), key=test_losses.__getitem__) + 1

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(epochs, train_losses, label="Train Loss", color="#008080", linewidth=2)
    axis.plot(epochs, test_losses, label="Test Loss", color="#E76F51", linewidth=2)
    axis.axvline(
        best_epoch,
        color="#00A896",
        linestyle="--",
        alpha=0.75,
        label=f"Best epoch: {best_epoch}",
    )
    axis.set_title("OdorPredictor Learning Curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("BCEWithLogitsLoss")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_data_loaders(
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, torch.utils.data.Subset]:
    """Wrap X/Y in TensorDataset, split 80/20, and create DataLoaders."""
    full_dataset = TensorDataset(features.float(), targets.float())
    train_size = int(TRAIN_RATIO * len(full_dataset))
    test_size = len(full_dataset) - train_size

    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=split_generator,
    )

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader, test_dataset


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one optimization epoch and return mean loss per sample."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_features, batch_targets in data_loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = criterion(logits, batch_targets)
        loss.backward()
        optimizer.step()

        batch_count = batch_features.shape[0]
        total_loss += loss.item() * batch_count
        total_samples += batch_count

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate without gradient tracking and return mean loss per sample."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)

            batch_count = batch_features.shape[0]
            total_loss += loss.item() * batch_count
            total_samples += batch_count

    return total_loss / total_samples


def print_random_top_3_prediction(
    model: nn.Module,
    test_dataset: torch.utils.data.Subset,
    label_names: list[str],
    device: torch.device,
) -> None:
    """Run inference on one random test sample and print its top three odors."""
    random_index = random.randrange(len(test_dataset))
    sample_features, _ = test_dataset[random_index]

    model.eval()
    with torch.no_grad():
        logits = model(sample_features.unsqueeze(0).to(device))
        probabilities = torch.sigmoid(logits).squeeze(0).cpu()

    top_probabilities, top_indices = torch.topk(probabilities, k=3)
    print(f"\nTop 3 dự đoán cho mẫu Test #{random_index}:")
    for rank, (probability, label_index) in enumerate(
        zip(top_probabilities, top_indices),
        start=1,
    ):
        label = label_names[label_index.item()]
        print(f"{rank}. {label}: {probability.item() * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the odor prediction model.")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("Số epoch phải lớn hơn 0.")
    if args.epochs > MAX_EPOCHS:
        raise ValueError(f"Số epoch tối đa là {MAX_EPOCHS}.")
    if args.batch_size <= 0:
        raise ValueError("Batch size phải lớn hơn 0.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Giữ Y/label_names từ TensorDataset, nhưng tái tạo X để bật chirality thật sự.
    dataset = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    stored_X, Y = dataset.tensors
    label_names = list(dataset.label_names)
    validate_loaded_data(stored_X, Y, label_names)
    del stored_X, dataset

    print(f"Đang tạo lại Morgan Fingerprints có chirality từ {SMILES_CSV_PATH}...")
    X = create_chiral_features_from_csv(
        SMILES_CSV_PATH,
        expected_rows=Y.shape[0],
    )
    validate_loaded_data(X, Y, label_names)

    device = choose_device()
    print(f"Device: {device}")
    print(f"X shape: {tuple(X.shape)} | Y shape: {tuple(Y.shape)}")
    print("Morgan config: radius=2 | nBits=2048 | useChirality=True")

    train_loader, test_loader, test_dataset = create_data_loaders(
        X,
        Y,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")

    model = OdorPredictor().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_test_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_loss_history: list[float] = []
    test_loss_history: list[float] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        test_loss = evaluate(model, test_loader, criterion, device)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}"
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
            print(
                f"Best Model mới tại epoch {epoch}: "
                f"Test Loss = {best_test_loss:.6f}"
            )
            print("Đã lưu mô hình thành công")
        else:
            epochs_without_improvement += 1
            print(
                "Test Loss không cải thiện: "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs"
            )

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early Stopping tại epoch {epoch}. "
                    f"Best Model ở epoch {best_epoch} "
                    f"với Test Loss = {best_test_loss:.6f}"
                )
                break

    plot_learning_curve(
        train_loss_history,
        test_loss_history,
        LEARNING_CURVE_PATH,
    )
    print(f"Đã lưu Learning Curve: {LEARNING_CURVE_PATH}")

    if best_epoch == 0:
        raise RuntimeError("Không có epoch hợp lệ để lưu Best Model.")
    best_state_dict = torch.load(
        BEST_WEIGHTS_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state_dict)
    print(
        f"Đã khôi phục Best Model từ epoch {best_epoch} "
        f"(Test Loss = {best_test_loss:.6f})"
    )

    print_random_top_3_prediction(model, test_dataset, label_names, device)


if __name__ == "__main__":
    main()
