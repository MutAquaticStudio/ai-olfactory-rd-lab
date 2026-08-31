"""Nền tảng Streamlit R&D hương liệu: Judge + Creator + Accord Scoring."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, Draw, rdDepictor, rdMolDescriptors
from torch import nn


APP_DIR = Path(__file__).resolve().parent
ODOR_DATASET_PATH = APP_DIR / "odor_morgan_tensor_dataset.pt"
JUDGE_WEIGHTS_PATH = APP_DIR / "odor_predictor_weights.pth"
CREATOR_WEIGHTS_PATH = APP_DIR / "smiles_creator_weights.pth"
VOCAB_PATH = APP_DIR / "smiles_vocab.json"
SMILES_DATASET_PATH = APP_DIR / "clean_dataset.csv"

FINGERPRINT_SIZE = 2048
ODOR_LABEL_COUNT = 113
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
PAD_TOKEN = "<PAD>"
END_TOKEN = "<END>"
DEFAULT_SMILES = "CCCCC1C(CC(=O)C1)CC(=O)OC"
NOVEL_POOL_SIZE = 5
MAX_ATTEMPTS = 200


def select_device() -> torch.device:
    """Ưu tiên Apple Metal; CPU là phương án dự phòng ổn định."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class OdorPredictor(nn.Module):
    """Mô hình Judge đa nhãn: 2048 -> 1024 -> 512 -> 113."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FINGERPRINT_SIZE, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, ODOR_LABEL_COUNT),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class SMILES_LSTM(nn.Module):
    """Char-LSTM Creator khớp hoàn toàn với kiến trúc huấn luyện."""

    def __init__(self, vocab_size: int, pad_idx: int) -> None:
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
        return self.output(recurrent_output), hidden


@dataclass(frozen=True)
class RankedCandidate:
    """Dữ liệu gọn của một Candidate R&D đã được Judge chấm điểm."""

    isomeric_smiles: str
    canonical_smiles: str
    accord_score: float
    target_probabilities: Tuple[Tuple[str, float], ...]
    secondary_odors: Tuple[Tuple[str, float], ...]
    molecular_formula: str
    exact_molecular_weight: float
    note_class: str
    log_p: float


@st.cache_resource(show_spinner=False)
def load_judge() -> Tuple[OdorPredictor, Tuple[str, ...]]:
    """Nạp một lần mô hình Judge và thứ tự 113 nhãn mùi."""
    dataset = torch.load(
        ODOR_DATASET_PATH,
        map_location="cpu",
        weights_only=False,
    )
    label_names = tuple(str(label) for label in dataset.label_names)
    if len(label_names) != ODOR_LABEL_COUNT:
        raise ValueError(
            f"Dataset phải có {ODOR_LABEL_COUNT} nhãn, "
            f"nhưng nhận được {len(label_names)}."
        )

    model = OdorPredictor()
    state_dict = torch.load(
        JUDGE_WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(select_device()).eval()
    return model, label_names


@st.cache_resource(show_spinner=False)
def load_creator() -> Tuple[SMILES_LSTM, Dict[str, int], Tuple[str, ...]]:
    """Nạp một lần Creator cùng vocabulary hai chiều."""
    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    char_to_idx = {
        str(token): int(index)
        for token, index in vocabulary["char_to_idx"].items()
    }
    idx_to_char = tuple(str(token) for token in vocabulary["idx_to_char"])
    if PAD_TOKEN not in char_to_idx or END_TOKEN not in char_to_idx:
        raise ValueError("Vocabulary phải chứa <PAD> và <END>.")
    if len(char_to_idx) != len(idx_to_char):
        raise ValueError("Hai chiều ánh xạ vocabulary không cùng kích thước.")
    for token, index in char_to_idx.items():
        if index >= len(idx_to_char) or idx_to_char[index] != token:
            raise ValueError("Vocabulary có ánh xạ char/index không nhất quán.")

    model = SMILES_LSTM(
        vocab_size=len(idx_to_char),
        pad_idx=char_to_idx[PAD_TOKEN],
    )
    state_dict = torch.load(
        CREATOR_WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(select_device()).eval()
    return model, char_to_idx, idx_to_char


@st.cache_data(show_spinner=False)
def load_existing_smiles_set() -> Set[str]:
    """Canonical hóa dữ liệu gốc để tra cứu trùng lặp trung bình O(1)."""
    dataframe = pd.read_csv(SMILES_DATASET_PATH)
    smiles_column = next(
        (column for column in dataframe.columns if column.lower() == "smiles"),
        None,
    )
    if smiles_column is None:
        raise ValueError("clean_dataset.csv không có cột SMILES.")

    existing_smiles_set: Set[str] = set()
    with rdBase.BlockLogs():
        for value in dataframe[smiles_column].dropna():
            molecule = Chem.MolFromSmiles(str(value).strip())
            if molecule is not None:
                existing_smiles_set.add(
                    Chem.MolToSmiles(molecule, canonical=True)
                )

    if not existing_smiles_set:
        raise ValueError("Không nạp được SMILES hợp lệ từ clean_dataset.csv.")
    return existing_smiles_set


def model_device(model: nn.Module) -> torch.device:
    """Lấy thiết bị hiện đang giữ tham số mô hình."""
    return next(model.parameters()).device


def create_morgan_tensor(molecule: Chem.Mol) -> torch.Tensor:
    """Morgan radius=2, 2048 bit, bắt buộc giữ thông tin chirality."""
    # Dùng đúng API được yêu cầu; BlockLogs chỉ chặn cảnh báo deprecation của RDKit.
    with rdBase.BlockLogs():
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius=2,
            nBits=FINGERPRINT_SIZE,
            useChirality=True,
        )
    fingerprint_array = np.zeros((FINGERPRINT_SIZE,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    return torch.from_numpy(fingerprint_array)


def predict_probabilities(
    model: OdorPredictor,
    molecules: Sequence[Chem.Mol],
) -> torch.Tensor:
    """Chấm một batch phân tử và trả xác suất trên CPU."""
    if not molecules:
        return torch.empty((0, ODOR_LABEL_COUNT), dtype=torch.float32)
    features = torch.stack(
        [create_morgan_tensor(molecule) for molecule in molecules]
    ).to(model_device(model))
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(features))
    return probabilities.cpu()


def top_odors(
    probabilities: torch.Tensor,
    label_names: Sequence[str],
    count: int,
    excluded_indices: Optional[Set[int]] = None,
) -> Tuple[Tuple[str, float], ...]:
    """Lấy các nốt hương cao nhất, có thể loại toàn bộ target đã chọn."""
    scores = probabilities.clone()
    for index in excluded_indices or set():
        scores[index] = float("-inf")
    k = min(count, scores.numel() - len(excluded_indices or set()))
    values, indices = torch.topk(scores, k=max(k, 0))
    return tuple(
        (label_names[index.item()], float(value.item()))
        for value, index in zip(values, indices)
    )


def smiles_representations(molecule: Chem.Mol) -> Tuple[str, str]:
    """Trả Isomeric SMILES và Canonical SMILES không mang stereo."""
    isomeric_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )
    return isomeric_smiles, canonical_smiles


def prepare_molecule_for_drawing(molecule: Chem.Mol) -> Chem.Mol:
    """Tạo bản sao 2D và gán Wedge/Dash cho các tâm lập thể đã biết."""
    drawing_molecule = Chem.Mol(molecule)
    Chem.AssignStereochemistry(drawing_molecule, cleanIt=True, force=True)
    rdDepictor.Compute2DCoords(drawing_molecule, canonOrient=True)
    if drawing_molecule.GetNumConformers():
        Chem.WedgeMolBonds(
            drawing_molecule,
            drawing_molecule.GetConformer(),
        )
    return drawing_molecule


def molecule_image(molecule: Chem.Mol, size: Tuple[int, int] = (720, 480)):
    """Render ảnh PIL rõ nét, nền trắng, giữ liên kết Wedge/Dash."""
    drawing_molecule = prepare_molecule_for_drawing(molecule)
    return Draw.MolToImage(
        drawing_molecule,
        size=size,
        kekulize=True,
        wedgeBonds=True,
    )


def sample_smiles_string(
    model: SMILES_LSTM,
    char_to_idx: Dict[str, int],
    idx_to_char: Sequence[str],
    temperature: float,
    start_str: str = "C",
    max_len: int = 60,
) -> str:
    """Sinh tuần tự một chuỗi ký tự SMILES theo temperature."""
    if temperature <= 0:
        raise ValueError("Temperature phải lớn hơn 0.")
    unknown_characters = set(start_str) - set(char_to_idx)
    if unknown_characters:
        raise ValueError(
            "Ký tự mở đầu không có trong vocabulary: "
            + ", ".join(sorted(unknown_characters))
        )

    device = model_device(model)
    token_ids = torch.tensor(
        [[char_to_idx[character] for character in start_str]],
        dtype=torch.long,
        device=device,
    )
    generated = start_str
    pad_idx = char_to_idx[PAD_TOKEN]
    end_idx = char_to_idx[END_TOKEN]

    with torch.inference_mode():
        logits, hidden = model(token_ids)
        for _ in range(max_len - len(start_str)):
            sampling_logits = logits[0, -1] / temperature
            sampling_logits[pad_idx] = float("-inf")
            probabilities = torch.softmax(sampling_logits, dim=-1)
            next_idx = int(torch.multinomial(probabilities, num_samples=1).item())
            if next_idx == end_idx:
                break

            next_character = idx_to_char[next_idx]
            generated += next_character
            next_token = torch.tensor(
                [[next_idx]],
                dtype=torch.long,
                device=device,
            )
            logits, hidden = model(next_token, hidden)

    return generated


def collect_novel_molecules(
    model: SMILES_LSTM,
    char_to_idx: Dict[str, int],
    idx_to_char: Sequence[str],
    temperature: float,
    existing_smiles_set: Set[str],
    required_count: int = NOVEL_POOL_SIZE,
    max_attempts: int = MAX_ATTEMPTS,
) -> Tuple[List[Chem.Mol], int]:
    """Auto-retry đến đủ phân tử hợp lệ, novel và không trùng trong batch."""
    novel_molecules: List[Chem.Mol] = []
    generated_smiles_set: Set[str] = set()
    attempts = 0

    while len(novel_molecules) < required_count and attempts < max_attempts:
        attempts += 1
        generated_smiles = sample_smiles_string(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            temperature=temperature,
        )
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(generated_smiles)
        if molecule is None:
            continue

        # RDKit mặc định giữ stereo trong khóa canonical dùng để lọc trùng.
        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        if (
            canonical_smiles in existing_smiles_set
            or canonical_smiles in generated_smiles_set
        ):
            continue

        generated_smiles_set.add(canonical_smiles)
        novel_molecules.append(molecule)

    return novel_molecules, attempts


def classify_note(exact_molecular_weight: float) -> str:
    """Phân tầng hương theo ngưỡng MW trong đặc tả."""
    if exact_molecular_weight < 150:
        return "Hương đầu (Top Note)"
    if exact_molecular_weight <= 220:
        return "Hương giữa (Middle Note)"
    return "Hương cuối (Base Note)"


def geometric_mean(probabilities: torch.Tensor) -> float:
    """Geometric mean ổn định số học trong miền log."""
    safe_probabilities = probabilities.clamp_min(1e-12)
    return float(torch.exp(torch.log(safe_probabilities).mean()).item())


def rank_novel_candidates(
    judge_model: OdorPredictor,
    label_names: Sequence[str],
    target_odors: Sequence[str],
    molecules: Sequence[Chem.Mol],
) -> List[RankedCandidate]:
    """Tính Accord Score đa mục tiêu và xếp hạng Candidate."""
    if not target_odors or not molecules:
        return []

    label_to_index = {label: index for index, label in enumerate(label_names)}
    target_indices = [label_to_index[label] for label in target_odors]
    target_index_set = set(target_indices)
    probability_matrix = predict_probabilities(judge_model, molecules)
    candidates: List[RankedCandidate] = []

    for molecule, probabilities in zip(molecules, probability_matrix):
        target_scores = probabilities[target_indices]
        isomeric_smiles, canonical_smiles = smiles_representations(molecule)
        exact_molecular_weight = float(Descriptors.ExactMolWt(molecule))
        candidates.append(
            RankedCandidate(
                isomeric_smiles=isomeric_smiles,
                canonical_smiles=canonical_smiles,
                accord_score=geometric_mean(target_scores),
                target_probabilities=tuple(
                    (label, float(probabilities[index].item()))
                    for label, index in zip(target_odors, target_indices)
                ),
                secondary_odors=top_odors(
                    probabilities,
                    label_names,
                    count=3,
                    excluded_indices=target_index_set,
                ),
                molecular_formula=rdMolDescriptors.CalcMolFormula(molecule),
                exact_molecular_weight=exact_molecular_weight,
                note_class=classify_note(exact_molecular_weight),
                log_p=float(Descriptors.MolLogP(molecule)),
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: candidate.accord_score,
        reverse=True,
    )


def probability_percent(probability: float) -> float:
    """Giới hạn xác suất về miền hợp lệ và đổi sang phần trăm."""
    return max(0.0, min(100.0, probability * 100.0))


def render_probability_bar(label: str, probability: float) -> None:
    """Hiển thị nhãn, phần trăm và thanh tiến trình Teal."""
    percent = probability_percent(probability)
    safe_label = html.escape(str(label))
    st.markdown(
        (
            '<div class="probability-row">'
            f'<span>{safe_label}</span><strong>{percent:.1f}%</strong>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.progress(int(round(percent)))


def apply_page_style() -> None:
    """Áp dụng hệ màu Teal/White và quy tắc responsive."""
    st.markdown(
        """
        <style>
        :root {
            --teal: #008080;
            --mint: #00A896;
            --ice: #F4F9F9;
            --ink: #153B3B;
            --muted: #557070;
            --line: #CFE4E2;
            --white: #FFFFFF;
        }

        .stApp {
            background: #FFFFFF;
            color: var(--ink);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4, [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: var(--teal) !important;
            letter-spacing: -0.02em;
        }

        h1, [data-testid="stMarkdownContainer"] h1 {
            text-align: center;
        }

        .button-spacer {
            height: 1.78rem;
        }

        [data-baseweb="tab-list"] {
            background: #EAF5F4;
            border: 1px solid var(--line);
            border-radius: 14px;
            gap: 0.35rem;
            padding: 0.35rem;
        }

        [data-baseweb="tab"] {
            border-radius: 10px;
            color: var(--muted);
            flex: 1 1 0;
            font-weight: 750;
            justify-content: center;
            min-height: 44px;
        }

        [aria-selected="true"][data-baseweb="tab"] {
            background: var(--white);
            color: var(--teal);
            box-shadow: 0 4px 14px rgba(0, 128, 128, 0.10);
        }

        [data-baseweb="tab-highlight"] {
            background-color: var(--teal) !important;
        }

        [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"]) {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line) !important;
            border-radius: 18px;
            box-shadow: 0 12px 34px rgba(0, 95, 95, 0.07);
        }

        [data-testid="stTextInput"] input,
        [data-baseweb="select"] > div {
            background: #FFFFFF;
            border-color: #B9D9D6;
            color: var(--ink);
        }

        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p {
            color: var(--ink) !important;
            font-weight: 750 !important;
        }

        [data-testid="stTextInputRootElement"] {
            border-color: #9FCFCB !important;
        }

        [data-baseweb="tag"] {
            background-color: #DDF3F0 !important;
            color: #006E6E !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--teal), var(--mint));
            border: 0;
            border-radius: 11px;
            box-shadow: 0 8px 20px rgba(0, 128, 128, 0.18);
            color: white;
            font-weight: 800;
            min-height: 46px;
            width: 100%;
        }

        .stButton > button:hover {
            box-shadow: 0 10px 24px rgba(0, 128, 128, 0.28);
            color: white;
            transform: translateY(-1px);
        }

        [data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, var(--teal), var(--mint));
        }

        [data-testid="stImage"] img {
            background: #FFFFFF;
            border: 1px solid #D9EAE8;
            border-radius: 14px;
            padding: 0.55rem;
        }

        [data-testid="stCode"] {
            border: 1px solid #D6E8E6;
            border-radius: 10px;
        }

        [data-testid="stCode"] pre {
            background: #F7FBFB !important;
            color: #153B3B !important;
        }

        .probability-row {
            align-items: center;
            color: var(--ink);
            display: flex;
            font-size: 0.94rem;
            justify-content: space-between;
            margin: 0.35rem 0 0.2rem;
        }

        .probability-row strong {
            color: var(--teal);
        }

        .candidate-header {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            justify-content: space-between;
            margin-bottom: 0.8rem;
        }

        .candidate-rank {
            color: var(--teal);
            font-size: 1.12rem;
            font-weight: 850;
        }

        .candidate-badges {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .novel-badge {
            background: #DDF3F0;
            border: 1px solid #9ED9D2;
            border-radius: 999px;
            color: #006D68;
            font-size: 0.82rem;
            font-weight: 800;
            padding: 0.4rem 0.72rem;
        }

        .accord-badge {
            background: var(--teal);
            border-radius: 999px;
            color: #FFFFFF;
            font-size: 0.82rem;
            font-weight: 850;
            padding: 0.42rem 0.76rem;
        }

        .descriptor-grid {
            display: grid;
            gap: 0.55rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 0.75rem;
        }

        .descriptor-item {
            background: var(--ice);
            border: 1px solid #D8EAE8;
            border-radius: 10px;
            min-height: 76px;
            padding: 0.65rem 0.7rem;
        }

        .descriptor-label {
            color: var(--muted);
            display: block;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }

        .descriptor-value {
            color: var(--ink);
            display: block;
            font-size: 0.92rem;
            font-weight: 800;
            overflow-wrap: anywhere;
        }

        .note-class {
            color: var(--teal);
            display: block;
            font-size: 0.74rem;
            font-weight: 750;
            margin-top: 0.18rem;
        }

        .secondary-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.35rem;
        }

        .secondary-pill {
            background: #EDF7F6;
            border: 1px solid #D0E8E5;
            border-radius: 999px;
            color: #315F5D;
            font-size: 0.83rem;
            font-weight: 700;
            padding: 0.35rem 0.6rem;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1.25rem;
            }

            h1 {
                font-size: 1.82rem !important;
            }

            [data-baseweb="tab-list"] {
                align-items: stretch;
                flex-direction: column;
            }

            [data-baseweb="tab"] {
                justify-content: flex-start;
                width: 100%;
            }

            .descriptor-grid {
                grid-template-columns: 1fr;
            }

            .button-spacer {
                height: 0;
            }

            .candidate-header,
            .candidate-badges {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_judge_tab(
    judge_model: OdorPredictor,
    label_names: Sequence[str],
) -> None:
    """Tab phân tích cấu trúc và hồ sơ mùi của một phân tử."""
    with st.container(border=True):
        input_column, action_column = st.columns([3.3, 1.0], gap="medium")
        with input_column:
            input_smiles = st.text_input(
                "Nhập chuỗi SMILES",
                value=DEFAULT_SMILES,
                key="judge_smiles",
                help="Có thể dùng ký hiệu @, @@, / hoặc \\ để mô tả đồng phân.",
            ).strip()
        with action_column:
            st.markdown('<div class="button-spacer"></div>', unsafe_allow_html=True)
            analyze_clicked = st.button(
                "Dự đoán Mùi hương",
                type="primary",
                key="judge_button",
            )

    if not analyze_clicked:
        st.info("Nhập SMILES và bấm “Dự đoán Mùi hương” để bắt đầu phân tích.")
        return

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(input_smiles)
    if molecule is None:
        st.error("Chuỗi SMILES không hợp lệ, vui lòng kiểm tra lại!")
        return

    probabilities = predict_probabilities(judge_model, [molecule])[0]
    predictions = top_odors(probabilities, label_names, count=5)
    isomeric_smiles, canonical_smiles = smiles_representations(molecule)

    left_column, right_column = st.columns([1.05, 1.0], gap="large")
    with left_column:
        st.markdown("#### Cấu trúc 2D & Lập thể")
        st.image(
            molecule_image(molecule),
            width="stretch",
            caption="Wedge/Dash thể hiện tâm lập thể khi SMILES có khai báo stereo.",
        )

    with right_column:
        st.markdown("#### Chuẩn hóa Cấu trúc")
        st.caption("Isomeric SMILES")
        st.code(isomeric_smiles, language=None)
        st.caption("Canonical SMILES")
        st.code(canonical_smiles, language=None)
        st.markdown("#### Top 5 Nốt hương dự đoán")
        for label, probability in predictions:
            render_probability_bar(label, probability)


def render_descriptor_grid(candidate: RankedCandidate) -> None:
    """Khối thông số hóa lý gọn cho Candidate R&D."""
    formula = html.escape(candidate.molecular_formula)
    note_class = html.escape(candidate.note_class)
    st.markdown(
        f"""
        <div class="descriptor-grid">
            <div class="descriptor-item">
                <span class="descriptor-label">Công thức hóa học</span>
                <span class="descriptor-value">{formula}</span>
            </div>
            <div class="descriptor-item">
                <span class="descriptor-label">Exact MW</span>
                <span class="descriptor-value">{candidate.exact_molecular_weight:.3f} Da</span>
                <span class="note-class">{note_class}</span>
            </div>
            <div class="descriptor-item">
                <span class="descriptor-label">LogP</span>
                <span class="descriptor-value">{candidate.log_p:.3f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_candidate(candidate: RankedCandidate, rank: int) -> None:
    """Render một thẻ Candidate R&D đầy đủ cấu trúc, hóa lý và mùi."""
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(candidate.isomeric_smiles)
    if molecule is None:
        return

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="candidate-header">
                <span class="candidate-rank">Candidate #{rank}</span>
                <div class="candidate-badges">
                    <span class="novel-badge">✨ 100% Novel Captive</span>
                    <span class="accord-badge">Accord Score: {candidate.accord_score * 100:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left_column, right_column = st.columns([1.0, 1.65], gap="large")
        with left_column:
            st.image(
                molecule_image(molecule, size=(640, 460)),
                width="stretch",
                caption="2D Render • Wedge/Dash stereochemistry",
            )

        with right_column:
            st.caption("Isomeric SMILES")
            st.code(candidate.isomeric_smiles, language=None)
            st.caption("Canonical SMILES")
            st.code(candidate.canonical_smiles, language=None)
            st.markdown("##### Thông số Hóa lý R&D")
            render_descriptor_grid(candidate)

        st.divider()
        st.markdown("#### Olfactory Profile")
        st.caption("Xác suất của các Nốt hương Mục tiêu")
        for label, probability in candidate.target_probabilities:
            render_probability_bar(label, probability)

        st.caption("Top 3 Nốt hương phụ")
        secondary_pills = "".join(
            (
                '<span class="secondary-pill">'
                f"{html.escape(label)} · {probability_percent(probability):.1f}%"
                "</span>"
            )
            for label, probability in candidate.secondary_odors
        )
        st.markdown(
            f'<div class="secondary-list">{secondary_pills}</div>',
            unsafe_allow_html=True,
        )


def render_creator_tab(
    judge_model: OdorPredictor,
    creator_model: SMILES_LSTM,
    char_to_idx: Dict[str, int],
    idx_to_char: Sequence[str],
    label_names: Sequence[str],
    existing_smiles_set: Set[str],
) -> None:
    """Tab sinh Candidate novel, chấm Accord Score và trình bày Top 3."""
    preferred_defaults = [
        label for label in ("jasmine", "woody") if label in label_names
    ]
    if not preferred_defaults:
        preferred_defaults = [label_names[0]]

    with st.container(border=True):
        target_column, temperature_column, action_column = st.columns(
            [1.45, 1.0, 1.2],
            gap="large",
        )
        with target_column:
            target_odors = st.multiselect(
                "Nốt hương Mục tiêu (Target Odors)",
                options=list(label_names),
                default=preferred_defaults,
                help="Chọn một hoặc nhiều nốt; Accord Score là geometric mean.",
            )
        with temperature_column:
            temperature = st.slider(
                "Mức độ sáng tạo (Temperature)",
                min_value=0.2,
                max_value=1.2,
                value=0.8,
                step=0.1,
                help="Thấp hơn: an toàn hơn. Cao hơn: đa dạng hơn nhưng dễ lỗi hóa trị.",
            )
        with action_column:
            st.markdown('<div class="button-spacer"></div>', unsafe_allow_html=True)
            create_clicked = st.button(
                "🔮 Generative New Captive Molecules",
                type="primary",
                key="creator_button",
            )

    if not create_clicked:
        st.info(
            "Chọn hợp âm mục tiêu và khởi chạy quy trình sinh–lọc–chấm điểm tự động."
        )
        return
    if not target_odors:
        st.error("Vui lòng chọn ít nhất một Nốt hương Mục tiêu.")
        return

    with st.spinner(
        "Đang tự động sinh, lọc trùng và chấm điểm Candidate R&D..."
    ):
        novel_molecules, attempts = collect_novel_molecules(
            model=creator_model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            temperature=temperature,
            existing_smiles_set=existing_smiles_set,
            required_count=NOVEL_POOL_SIZE,
            max_attempts=MAX_ATTEMPTS,
        )
        ranked_candidates = rank_novel_candidates(
            judge_model=judge_model,
            label_names=label_names,
            target_odors=target_odors,
            molecules=novel_molecules,
        )

    if len(novel_molecules) < NOVEL_POOL_SIZE:
        st.warning(
            f"Đã chạm giới hạn {MAX_ATTEMPTS} lần thử và thu được "
            f"{len(novel_molecules)}/{NOVEL_POOL_SIZE} phân tử novel hợp lệ. "
            "Hãy thử điều chỉnh Temperature."
        )
    else:
        st.success(
            f"Đã thu thập đủ {NOVEL_POOL_SIZE} phân tử novel hợp lệ sau "
            f"{attempts} lần thử và chọn Top 3 theo Accord Score."
        )

    if not ranked_candidates:
        st.error("Chưa sinh được Candidate hợp lệ trong lần chạy này.")
        return

    st.markdown("### Top 3 Candidate R&D")
    for rank, candidate in enumerate(ranked_candidates[:3], start=1):
        render_candidate(candidate, rank)


def main() -> None:
    """Điểm vào của ứng dụng Streamlit."""
    st.set_page_config(
        page_title="AI Olfactory R&D Lab",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_page_style()

    st.title("🧪 AI Olfactory R&D Lab")

    try:
        judge_model, label_names = load_judge()
        creator_model, char_to_idx, idx_to_char = load_creator()
        # Tên biến giữ đúng vai trò tra cứu novelty O(1) trong Master Prompt.
        existing_smiles_set = load_existing_smiles_set()
    except Exception as error:
        st.error(f"Không thể nạp tài nguyên ứng dụng: {error}")
        st.stop()

    judge_tab, creator_tab = st.tabs(
        [
            "🔍 Phân tích Phân tử (The Judge)",
            "🔮 Sáng tạo Captive Mới (Creator + Judge)",
        ]
    )

    with judge_tab:
        render_judge_tab(judge_model, label_names)

    with creator_tab:
        render_creator_tab(
            judge_model=judge_model,
            creator_model=creator_model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            label_names=label_names,
            existing_smiles_set=existing_smiles_set,
        )


if __name__ == "__main__":
    main()
