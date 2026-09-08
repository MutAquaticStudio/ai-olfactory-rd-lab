# CROWN — encoding đã giải quyết, training chưa qua data gate

**Cập nhật:** người dùng sau đó yêu cầu training thử nghiệm và xuất curve.
Run riêng đã hoàn tất, không thay đổi gate/release ở báo cáo này:
[kết quả exploratory](CROWN_EXPLORATORY_TRAINING.md).

Ngày: 08/09/2026. Đây là technical review dưới yêu cầu của người dùng, không
phải xác nhận của tác giả hay chữ ký chuyên gia cảm quan. Không chạy optimizer,
không thay registry hoặc production weights.

## 1. Encoding

Kết luận kỹ thuật: **1 = yes, 0 = no, ô trống = UNASSESSED**, áp dụng ở cấp
phản hồi participant–odor–occasion. Căn cứ phối hợp:

- [Bài báo, Technical Validation / Validity và Figure 4](https://www.nature.com/articles/s41597-025-04644-2)
  mô tả thống kê là tỷ lệ người trả lời yes cho descriptor.
- [Notebook tác giả v3](https://zenodo.org/records/15657278), cells 21/26
  (zero-based), tính thống kê đó bằng trung bình trực tiếp các mã nhị phân.

Với biến chỉ có 0/1, trung bình bằng tỷ lệ mã 1. Vì bài gọi đại lượng đó là
tỷ lệ yes, mã 1 tương ứng yes. Đây là suy luận đối chiếu paper/code rõ căn cứ,
không phải trích dẫn một numeric codebook hoặc thư tác giả.

Kiểm tra hash notebook và giữ policy có version trong
`data/crown_descriptor_policy_v1.json`. Kết luận này cập nhật trạng thái
pending của audit trước; artifact cũ không bị sửa.

212.160 vị trí phản hồi nguồn được giữ nguyên:

| Trạng thái cấp trial | Số lượng |
|---|---:|
| PRESENT_AT_TRIAL | 40.292 |
| ABSENT_AT_TRIAL | 165.386 |
| UNASSESSED | 6.482 |

Các số trên gồm cả cohort/excluded của nguồn, không phải số training targets,
cũng không phải số phân tử độc lập. Training `presence_state` vẫn UNASSESSED
cho đến khi có release được duyệt; `intensity` vẫn null.

## 2. Mapping đã khóa ở cấp nguồn

10 mapping: sweet→sweet, sour→sour, fruit→fruity, spices→spicy,
garlic→garlic, fish→fishy, burnt→burnt, grass→grassy, wood→woody,
flower→floral. Không suy rộng thành jasmine/citrus hoặc các descriptor cụ thể.

6 descriptor giữ source-only: bakery, decayed, chemical, musky, sweaty,
ammonia/urinous. Không đồng nhất lay-panel musky với perfumery musk.
Quyết định gồm actor/authorization/reason; `human_expert_approval=false`.

## 3. Điều kiện đo và cổng trước training

Main study chỉ có một occasion; không đạt yêu cầu hai phiên lặp độc lập
của release Judge hiện tại. Retest có 60 assessor ở hai occasion, sáu mã mùi.
Không biến 60 assessor thành 60 phân tử hoặc trộn patient với healthy.

Tính Krippendorff's alpha nominal với unit = molecule × occasion, cột = assessor;
không điền missing bằng zero. Alpha này đánh giá đồng thuận giữa assessor
trên các unit, không phải Spearman test–retest hoặc ICC của intensity.
Các main cohort/set được tính riêng; bảng dưới là nhóm retest, chỉ mang nghĩa
descriptive audit khi identity/context chưa được giải quyết đầy đủ.

| Mapping target | Alpha retest | Ngưỡng hiện tại |
|---|---:|---:|
| sweet | 0.1480 | ≥0.50 |
| sour | 0.1049 | ≥0.50 |
| fruity | 0.1266 | ≥0.50 |
| spicy | 0.0932 | ≥0.50 |
| garlic | -0.0011 | ≥0.50 |
| fishy | 0.0017 | ≥0.50 |
| burnt | 0.0676 | ≥0.50 |
| grassy | 0.0226 | ≥0.50 |
| woody | 0.0244 | ≥0.50 |
| floral | 0.1072 | ≥0.50 |

**0/10 mapping qua cổng thống kê retest.** Đây không phải kết luận dữ liệu
không có giá trị; nó không đạt quy tắc consensus/repeated-panel đang khóa.
Không hạ ngưỡng alpha, bỏ yêu cầu replicate, hoặc thay mục tiêu học sang phân
phối phản hồi cá nhân chỉ để có một training run.

Các mục context/identity vẫn giữ hold theo audit trước: 35 dòng PEA không có
metadata cùng set; patient/retest chưa được gán context đã duyệt; 11 cờ stereo
RDKit và một conflict tên racemic/SMILES. CAS checksum 84/84 không đồng nghĩa
CAS/CID khớp với vật liệu đã ngửi. Không âm thầm sửa nguồn.

## 4. Artifact, kiểm thử và trạng thái training

Private output:
`~/.scent-molecule-studio/reviews/crown-2025-harmonized-source-v2`.

- `source_responses.parquet`: raw code + source presence + mapping, không target release.
- `policy.json`, `agreement.csv/json`, `training_preflight.json`.
- `condition_review.json`, `identity_review.json` giữ quyết định hold.
- `manifest.json`: SHA-256 cha/output/policy/script, version, timestamp.

Script `finalize_crown_review.py` kiểm tra checksum cha và sự thống nhất nguồn,
ghi vào thư mục mới nguyên tử, từ chối ghi đè. Không có network trong script.

Đã kiểm tra trực tiếp loader training hiện tại từ chối Parquet AUDIT_ONLY ở
cả strict mode và pre-panel mode. Đây là cổng dữ liệu được thực thi, không phải
training bị crash giữa epochs. **Chưa có model hoặc learning curve mới.**

70 tests liên quan qua trên Python 3.12, gồm fixture alpha=1, alpha=-0.5,
trường hợp không có variance, missing labels, mapping và mask. Hai warning
PyTorch về tối ưu nested tensor ở tests Creator có sẵn; không phải failure.

## 5. Quyết định còn cần trước retrain

Theo kế hoạch Judge hiện tại: bổ sung/review evidence đáp ứng repeat, agreement,
identity và điều kiện đo; sau đó mới release snapshot và split 60/10/15/15.

Nếu muốn train ngay để khảo sát, cần chấp thuận riêng một benchmark exploratory
với mục tiêu/consensus và giới hạn dữ liệu được khai báo rõ. Đó không phải
Judge đã vượt quality gate, không được promote và không thể gắn nhãn
calibrated/validated. Chưa chạy phương án này thay cho yêu cầu hiện tại.
