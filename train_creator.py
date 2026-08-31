"""Train a character-level LSTM that generates chemically valid SMILES."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from rdkit import Chem, rdBase
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


APP_DIR = Path(__file__).resolve().parent
CLEAN_DATASET_PATH = APP_DIR / "clean_dataset.csv"
VOCAB_PATH = APP_DIR / "smiles_vocab.json"
WEIGHTS_PATH = APP_DIR / "smiles_creator_weights.pth"

PAD_TOKEN = "<PAD>"
END_TOKEN = "<END>"

EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

TRAINING_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.002
GENERATION_TEMPERATURE = 0.8
GENERATION_SAMPLE_COUNT = 10


def select_device() -> torch.device:
    """Prefer Apple Metal acceleration and otherwise train on the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _clean_smiles_values(values: Sequence[object]) -> List[str]:
    """Convert a sequence to non-empty SMILES strings without deduplicating it."""
    smiles_list: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        smiles = str(value).strip()
        if smiles:
            smiles_list.append(smiles)
    return smiles_list


def load_smiles() -> List[str]:
    """Đọc trực tiếp toàn bộ danh sách SMILES từ clean_dataset.csv."""
    if not CLEAN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {CLEAN_DATASET_PATH}")

    dataframe = pd.read_csv(CLEAN_DATASET_PATH)
    smiles_column = next(
        (column for column in dataframe.columns if column.lower() == "smiles"),
        None,
    )
    if smiles_column is None:
        raise ValueError(f"{CLEAN_DATASET_PATH.name} không có cột SMILES.")

    smiles_list = _clean_smiles_values(dataframe[smiles_column].tolist())
    if not smiles_list:
        raise ValueError(f"Không có SMILES hợp lệ trong {CLEAN_DATASET_PATH.name}.")

    print(
        f"Đã nạp {len(smiles_list)} chuỗi SMILES từ "
        f"{CLEAN_DATASET_PATH.name}."
    )
    return smiles_list


def build_vocabulary(
    smiles_list: Sequence[str],
) -> Tuple[Dict[str, int], List[str]]:
    """Build deterministic token-index mappings for all SMILES characters."""
    characters = sorted({character for smiles in smiles_list for character in smiles})
    idx_to_char = [PAD_TOKEN, END_TOKEN, *characters]
    char_to_idx = {token: index for index, token in enumerate(idx_to_char)}
    return char_to_idx, idx_to_char


def save_vocabulary(
    char_to_idx: Dict[str, int],
    idx_to_char: Sequence[str],
) -> None:
    """Persist both vocabulary mappings for reuse by the Streamlit app."""
    vocabulary = {
        "char_to_idx": char_to_idx,
        "idx_to_char": list(idx_to_char),
        "pad_token": PAD_TOKEN,
        "end_token": END_TOKEN,
    }
    with VOCAB_PATH.open("w", encoding="utf-8") as file:
        json.dump(vocabulary, file, ensure_ascii=False, indent=2)
    print(f"Đã lưu bộ từ vựng ({len(idx_to_char)} token): {VOCAB_PATH.name}")


class SMILESDataset(Dataset):
    """Create next-character training pairs from raw SMILES strings."""

    def __init__(self, smiles_list: Sequence[str], char_to_idx: Dict[str, int]):
        self.smiles_list = list(smiles_list)
        self.char_to_idx = char_to_idx
        self.end_idx = char_to_idx[END_TOKEN]

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.char_to_idx[character] for character in self.smiles_list[index]]
        inputs = torch.tensor(encoded, dtype=torch.long)
        targets = torch.tensor(encoded[1:] + [self.end_idx], dtype=torch.long)
        return inputs, targets


def make_collate_fn(pad_idx: int):
    """Return a batch function that pads inputs and targets to equal lengths."""

    def collate_batch(
        batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = zip(*batch)
        padded_inputs = pad_sequence(
            inputs,
            batch_first=True,
            padding_value=pad_idx,
        )
        padded_targets = pad_sequence(
            targets,
            batch_first=True,
            padding_value=pad_idx,
        )
        return padded_inputs, padded_targets

    return collate_batch


class SMILES_LSTM(nn.Module):
    """Character-level recurrent model: Embedding -> LSTM -> vocabulary logits."""

    def __init__(self, vocab_size: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT,
        )
        self.output = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(token_ids)
        recurrent_output, hidden = self.lstm(embedded, hidden)
        logits = self.output(recurrent_output)
        return logits, hidden


def train_model(
    model: SMILES_LSTM,
    data_loader: DataLoader,
    device: torch.device,
    pad_idx: int,
    vocab_size: int,
    epochs: int = TRAINING_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> None:
    """Train the LSTM with teacher forcing and next-character prediction."""
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            valid_tokens = int((targets != pad_idx).sum().item())
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        average_loss = total_loss / max(total_tokens, 1)
        print(f"Epoch {epoch:02d}/{epochs} - Train Loss: {average_loss:.6f}")


def generate_smiles(
    model: SMILES_LSTM,
    char_to_idx: Dict[str, int],
    idx_to_char: Sequence[str],
    start_str: str = "C",
    max_len: int = 60,
    temperature: float = 0.8,
) -> Optional[str]:
    """Sample one SMILES and return its canonical form when RDKit accepts it."""
    if not start_str:
        raise ValueError("start_str không được để trống.")
    if max_len < len(start_str):
        raise ValueError("max_len phải lớn hơn hoặc bằng độ dài start_str.")
    if temperature <= 0:
        raise ValueError("temperature phải lớn hơn 0.")

    unknown_characters = sorted(set(start_str) - set(char_to_idx))
    if unknown_characters:
        raise ValueError(
            "start_str chứa ký tự ngoài vocabulary: "
            + ", ".join(repr(character) for character in unknown_characters)
        )

    device = next(model.parameters()).device
    prefix_ids = [char_to_idx[character] for character in start_str]
    generated = start_str
    pad_idx = char_to_idx[PAD_TOKEN]
    end_idx = char_to_idx[END_TOKEN]

    model.eval()
    with torch.no_grad():
        prefix_tensor = torch.tensor(
            [prefix_ids],
            dtype=torch.long,
            device=device,
        )
        logits, hidden = model(prefix_tensor)
        next_logits = logits[:, -1, :]

        while len(generated) < max_len:
            sampling_logits = next_logits.squeeze(0).clone()
            sampling_logits[pad_idx] = float("-inf")
            probabilities = torch.softmax(sampling_logits / temperature, dim=-1)
            next_idx = int(torch.multinomial(probabilities, num_samples=1).item())

            if next_idx == end_idx:
                break

            next_character = idx_to_char[next_idx]
            generated += next_character

            next_input = torch.tensor(
                [[next_idx]],
                dtype=torch.long,
                device=device,
            )
            logits, hidden = model(next_input, hidden)
            next_logits = logits[:, -1, :]

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(generated)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def parse_arguments() -> argparse.Namespace:
    """Parse runtime options that do not alter the fixed training contract."""
    parser = argparse.ArgumentParser(
        description="Train a character-level SMILES generator.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument("--start-str", default="C")
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.batch_size <= 0:
        raise ValueError("batch-size phải lớn hơn 0.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    smiles_list = load_smiles()
    char_to_idx, idx_to_char = build_vocabulary(smiles_list)
    save_vocabulary(char_to_idx, idx_to_char)

    pad_idx = char_to_idx[PAD_TOKEN]
    training_dataset = SMILESDataset(smiles_list, char_to_idx)
    data_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate_fn(pad_idx),
    )

    device = select_device()
    print(f"Thiết bị huấn luyện: {device}")
    model = SMILES_LSTM(vocab_size=len(idx_to_char), pad_idx=pad_idx)
    train_model(
        model=model,
        data_loader=data_loader,
        device=device,
        pad_idx=pad_idx,
        vocab_size=len(idx_to_char),
        epochs=TRAINING_EPOCHS,
        learning_rate=args.learning_rate,
    )

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Đã lưu trọng số mô hình: {WEIGHTS_PATH.name}")

    valid_smiles: List[str] = []
    seen_smiles = set()
    for _ in range(GENERATION_SAMPLE_COUNT):
        generated = generate_smiles(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            start_str=args.start_str,
            max_len=args.max_len,
            temperature=GENERATION_TEMPERATURE,
        )
        if generated is not None and generated not in seen_smiles:
            seen_smiles.add(generated)
            valid_smiles.append(generated)

    print(
        "\nCác SMILES hợp lệ "
        f"({len(valid_smiles)}/{GENERATION_SAMPLE_COUNT} lần sinh, "
        f"temperature={GENERATION_TEMPERATURE}):"
    )
    if valid_smiles:
        for index, smiles in enumerate(valid_smiles, start=1):
            print(f"{index:02d}. {smiles}")
    else:
        print("Không có chuỗi hợp lệ trong lần chạy này.")


if __name__ == "__main__":
    main()
