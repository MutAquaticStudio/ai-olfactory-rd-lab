# Judge PU: benchmark thử nghiệm 113 vị trí nhãn

## Trạng thái ngày 08/09/2026

Đã chạy đủ sáu cấu hình seed 42 và hai seed bổ sung cho cấu hình được chọn
trên validation. Không thay Judge/Creator production hoặc model registry.
Không chuyển snapshot `AUDIT_ONLY` thành dữ liệu được duyệt để train production.

Artifact local, không đưa vào Git:

`artifacts/judge/pu-113-20260908-release-retrain/`

Lượt retrain trước khi phát hành tái tạo cùng metrics của
`pu-113-20260908-sensitivity`; không chỉnh hyperparameter sau khi xem test.
[Learning curve được xuất bản](assets/judge-pu-113-release-learning-curve.png).

- 4.729 dòng nguồn; loại 1.854 dòng thiếu stereo xác định.
- 2.875 cấu trúc stereo-resolved; 13.118 ghi nhận positive.
- Train 1.703, calibration 296, validation 436, test 440 cấu trúc.
- 794 nhóm hóa học; không overlap connectivity hoặc Murcko scaffold giữa partition.
- 57 nhãn có ít nhất 10 connectivity positive độc lập trong train.
- 56 vị trí còn lại không đóng góp loss; score xuất ra là `NaN`, không phải zero.
- 429/440 cấu trúc test có positive trong tập 57 nhãn được đánh giá retrieval.
- MPS; lượt retrain phát hành khoảng 61,8 giây (lượt đầu 55,5 giây), thấp hơn ngân sách tối đa 12 giờ.
  Ngân sách là trần thời gian, không phải yêu cầu phải chạy đủ 12 giờ.

## Kết quả thực tế

Chọn cấu hình bằng **validation observed-positive recall@5**, không dùng test.

| Cấu hình | Epoch chạy | Best epoch | Validation R@5 | Test R@5 |
|---|---:|---:|---:|---:|
| Linear, prior offset 0, seed 42 | 100 | 100 | 0,5553 | 0,5493 |
| Linear, offset 0,05, seed 42 | 100 | 100 | 0,5654 | 0,5715 |
| Linear, offset 0,15, seed 42 | 71 | 51 | 0,5674 | 0,5580 |
| **MLP, offset 0, seed 42 — được chọn** | 100 | 100 | **0,5683** | **0,5687** |
| MLP, offset 0,05, seed 42 | 84 | 64 | 0,5562 | 0,5307 |
| MLP, offset 0,15, seed 42 | 49 | 29 | 0,5458 | 0,5141 |
| MLP, offset 0, seed 17 | 44 | 24 | 0,4871 | 0,4764 |
| MLP, offset 0, seed 23 | 100 | 99 | 0,5708 | 0,5614 |

Không đổi lựa chọn sang linear offset 0,05 chỉ vì nó có test score cao hơn.
Hai seed bổ sung dùng kiểm tra độ ổn định, không phải chọn seed tốt nhất trên test.

Model được chọn:

- Test recall@5 **0,5687**, recall@10 **0,7658**, MRR **0,8535**.
- Baseline tần suất nhãn từ train: recall@5 **0,4478**.
- Delta recall@5 so với baseline tần suất: **+0,1209**;
  bootstrap 95% CI theo chemical group **[+0,0684; +0,1776]**, 1.000 resamples.
- So với linear tốt nhất trên validation: delta **+0,0107**,
  CI **[-0,0073; +0,0251]** — chưa chứng minh MLP tốt hơn linear ở recall@5.
- Random ranking trên 57 nhãn có expected recall@5 **5/57 = 0,0877**;
  enrichment của model được chọn khoảng **6,48 lần**. Đây không phải xác suất đúng.

Recall@5 là trung bình tỷ lệ các positive đã biết được tìm lại trong top 5,
trên những cấu trúc có positive đủ điều kiện. Không phải “accuracy 56,87%”.
Các nhãn unknown có thể thật sự positive nhưng chưa được catalog ghi nhận.
Metric chỉ phản ánh khả năng truy xuất ghi nhận hiện có, dễ chịu ảnh hưởng
của nhãn phổ biến và thiên lệch lựa chọn trong catalog.

Seed 17 kém rõ rệt và prior sensitivity lớn cho thấy kết quả chưa ổn định.
Không promote model; không tuyên bố xác suất cảm quan đã calibration.

## Thiết kế khoa học

Implementation riêng: `train_pu_exploratory.py` và
`olfactory/training/pu_exploratory.py`. Không nới loader/gate Judge đã duyệt.

- `positive_observed=True` nghĩa là ghi nhận `PRESENT`; `False` nghĩa là chưa
  đánh giá, **không phải `ABSENT`**. Snapshot gốc và provenance không sửa.
- Kiểm tra SHA-256 CSV, Parquet/mapping và thứ tự 113 nhãn trước training.
- Canonical isomeric duplicates được gộp positive, giữ danh sách source row.
- Split 60/10/15/15, seed 42; connectivity, Murcko scaffold và Butina r2/2048,
  similarity 0,60 với chirality. Butina không bảo đảm mọi cặp xuyên partition
  có similarity dưới 0,60.
- Full-batch molecular training để có positive expectation cho nhãn hiếm.
  Marginal expectation dùng **tất cả train molecules**, gồm cả known positives.
- Loss: `pi * E_P[softplus(-f)] + max(0, E_X[softplus(f)] - pi * E_P[softplus(f)])`.
  Đây là nonnegative risk objective; không gán nhãn negative cho unknown.
- Prior: `r`, `r + 0.05*(1-r)`, `r + 0.15*(1-r)`; `r` chỉ tính từ train.
  Những giá trị này và giả định positive sampling đại diện chưa được xác minh.
- Adam lr 0,001, gradient norm cap 5, tối đa 100 epochs, patience 20;
  checkpoint theo validation recall@5. MLP 2048→1024→512→113, dropout 0,3.
- Calibration không fit; intensity, AP/Brier/ECE với assessed-negative evidence
  đều `NOT_EVALUABLE`. CROWN response-rate không được trộn vào task này.
- Test mang trạng thái `RETROSPECTIVE_EXPLORATORY`: dữ liệu từng được xem,
  không gọi là blind validation. Không so trực tiếp PU loss với BCE/MAE CROWN
  hay run catalog 254 output trước đây.

Phương pháp tham khảo: [Kiryo et al., Positive-Unlabeled Learning with
Non-Negative Risk Estimator, NeurIPS 2017](https://arxiv.org/abs/1703.00593).
Paper không xác minh class prior hoặc giả định sampling cho catalog này.

## Chạy và resume

```bash
cd /Users/nox/Documents/MD/CRAWL
.venv-training/bin/python train_pu_exploratory.py \
  --acknowledge-exploratory \
  --output artifacts/judge/pu-113-new-run
```

Resume dùng lại chính cấu hình, code và thư mục đó, thêm `--resume`.
Không ghi đè experiment đã tồn tại. Checkpoint ghi nguyên tử, chứa model,
optimizer, RNG, best weights và lịch sử. Deadline kiểm tra mỗi full batch;
run hết ngân sách ghi `BUDGET_EXHAUSTED`. Thời gian invocation bị crash được
tính bảo thủ vào ngân sách; không tự cấp lại 12 giờ khi resume.

Mỗi run có `learning_curve.png`, `history.csv/json`, `weights.pth`,
`checkpoint.pth`, `metrics.json` và `test_scores.npz`.
Toàn benchmark có `REPORT.md`, `selection.json`, `label_support.csv/json`,
`per_label_retrieval.csv/json`, `split.json`, manifest checksum và kiểm tra
production integrity. Sigmoid outputs gọi là `model_scores`.

## Kiểm thử

```bash
.venv-training/bin/python -m pytest tests/test_pu_exploratory.py -q
```

Bao gồm fixture nnPU độc lập, nonnegative correction, inactive gradient,
unknown/ABSENT distinction, snapshot checksum, stereo exclusion, duplicate
support, deterministic split/training, partition isolation, deadline,
atomic checkpoint crash/resume và bootstrap theo group.

Kiểm tra artifact thực tế đã đối chiếu lại checksum, membership đầy đủ,
không overlap scaffold/connectivity, best epoch, CPU/MPS score parity và
recall@5 bằng phép tính độc lập. Production binary/registry không thay đổi.

Để bước tiếp tới xác suất đáng tin cậy vẫn cần assessed negatives hợp lệ,
review/protocol/provenance, calibration partition đủ support và panel độc lập.
