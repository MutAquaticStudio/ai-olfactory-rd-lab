# CROWN: kết quả training thử nghiệm — 08/09/2026

Run: `artifacts/judge/crown-exploratory-20260908-s42`.
Training được thực hiện sau yêu cầu chạy thử nghiệm và xuất learning curve.
**Đã hoàn tất, nhưng chưa vượt baseline; không promote lên production.**

## Dữ liệu và mục tiêu học

- Nguồn private: `crown-2025-harmonized-source-v2`; checksum lưu trong config.
- Sau lọc: 8.510 dòng participant–odor, 69 nhóm điều kiện, **59 cấu trúc độc lập**.
- Mười descriptor: burnt, fishy, floral, fruity, garlic, grassy, sour, spicy,
  sweet, woody. 103 output còn lại bị mask; prediction export dùng NaN cho
  output không có dữ liệu, không trình bày giá trị ngẫu nhiên là dự đoán.
- Target = tỷ lệ yes quan sát được trong từng điều kiện, không phải nhãn
  consensus chắc chắn của phân tử. Ô trống không trở thành negative.
- Không gộp nồng độ/cohort thành cùng một target; loss cho mỗi ô
  condition–descriptor có trọng số như nhau. Do model chỉ nhận cấu trúc,
  nó **không có khả năng điều kiện hóa theo nồng độ**; các nhóm điều kiện là
  quan sát riêng, không giải quyết được giới hạn này của kiến trúc hiện tại.
- Loại patient/retest chưa rõ context, source-excluded, context không khớp,
  cờ stereo/identity và hai mã Tribut/Allycap có vấn đề vật liệu/tên theo nguồn.
- Các cấu trúc còn lại là unflagged source SMILES, chưa xác minh vật liệu
  thực nghiệm. Việc dùng cho exploratory không chuyển chúng thành evidence
  được human expert duyệt.

## Cấu hình

| Thông số | Giá trị |
|---|---|
| Model | Morgan MLP, khởi tạo mới |
| Kiến trúc | 2048 → 1024 → 512 → 113; ReLU; Dropout 0,3 |
| Morgan | Radius 2, 2048 bits, chirality bật |
| Loss | Masked unweighted BCE trên response rate |
| Optimizer | Adam, learning rate 0,001 |
| Batch size / seed | 64 / 42 |
| Device | Apple MPS |
| Max epochs / patience | 100 / 20 |
| Epoch đã chạy | 50, Early Stopping |
| Checkpoint tốt nhất | Epoch 30 |

Split chemical groups theo connectivity/Murcko/Butina, tỷ lệ mục tiêu
60/10/15/15. Mọi điều kiện của cùng connectivity ở cùng partition.

| Partition | Nhóm điều kiện | Cấu trúc duy nhất |
|---|---:|---:|
| Train | 42 | 32 |
| Calibration | 7 | 7 |
| Validation | 10 | 10 |
| Locked test | 10 | 10 |

Không có connectivity/scaffold group overlap. Calibration partition giữ
ngoài fitting/evaluation; không đủ cơ sở để gọi model calibrated. Locked test
chỉ được sử dụng sau khi chọn checkpoint bằng validation; các kết quả bên dưới
là exploratory retrospective, không dùng để thay hyperparameters của run này.

## Kết quả tại checkpoint tốt nhất

| Tập / phương pháp | BCE | MAE response rate |
|---|---:|---:|
| Model — Train | 0,385382 | 0,050234 |
| Model — Validation | 0,431067 | 0,079860 |
| Model — Locked test | 0,452215 | 0,118241 |
| Trung bình từng nhãn từ train — Locked test | **0,438968** | **0,103742** |

MAE test 0,118 tương đương khoảng 11,82 điểm phần trăm sai lệch response rate;
**không được diễn giải thành accuracy 88,18%**. Model hiện kém hơn baseline
trung bình train trên cả hai metric test. Mẫu test chỉ có 10 cấu trúc nên chưa
đủ để kết luận tổng quát hóa hoặc cải thiện độ chính xác Judge.

Curve gồm BCE và MAE train/validation theo epoch; các giá trị train được tính
ở evaluation mode (Dropout tắt) để so sánh nhất quán với validation. Không vẽ
test theo epoch hoặc dùng test để dừng sớm. Sau epoch 30, train tiếp tục giảm
nhưng validation BCE không đạt kỷ lục mới trong 20 epoch.

## Artifacts đã xuất

Trong thư mục run private ngoài Git:

- `learning_curve.png`: hai biểu đồ BCE và response-rate MAE.
- `history.csv`, `history.json`: toàn bộ 50 epoch.
- `weights.pth`: checkpoint **epoch 30**, không phải epoch cuối.
- `metrics.json`, `config.json`, `manifest.json`: số liệu và SHA-256.
- `conditions.parquet`, `targets.npz`, `split.json`, `exclusions.json`:
  dữ liệu thí nghiệm đã lọc, mask, membership và lý do loại.
- `predictions.npz`: output của checkpoint tốt nhất; nhãn chưa học để NaN.

Đã load lại weights bằng CPU và tái lập loss/prediction với tolerance 2e-6.
Đã kiểm tra checksum artifact, code, weights/dataset production và registry:
**production không đổi**. Không giảm các gate trong loader Judge hiện tại.

## Tái lập

```bash
.venv-training/bin/python train_crown_exploratory.py \
  --source "$HOME/.scent-molecule-studio/reviews/crown-2025-harmonized-source-v2" \
  --output artifacts/judge/crown-exploratory-20260908-s42 \
  --acknowledge-exploratory --max-epochs 100 --patience 20 --device auto
```

Output đã tồn tại sẽ bị từ chối; dùng tên run mới để tái lập. Muốn kiểm tra
deterministic CPU có thể chọn `--device cpu`; không cam kết bitwise-identical
giữa MPS và CPU. Không dùng kết quả test này để tuning rồi tiếp tục gọi cùng
test là untouched.
