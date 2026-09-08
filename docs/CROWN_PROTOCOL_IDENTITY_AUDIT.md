# CROWN — audit trước khi tạo nhãn training

**Cập nhật:** encoding và mapping đã được khóa qua đối chiếu paper/code;
xem [training preflight mới](CROWN_TRAINING_PREFLIGHT.md). Báo cáo dưới đây
ghi lại trạng thái audit trước khi có kết luận đó.

## Kết quả bước tiếp

Đã kiểm tra mã notebook tác giả, tạo công cụ audit offline, bảng mapping cần
review và đối chiếu stimulus theo cohort/set. **Chưa đủ điều kiện retrain**;
không thay production weights, không ghi record human approval thay người dùng.

Nguồn pin: [Zenodo 15657278, v3, CC BY 4.0](https://zenodo.org/records/15657278).
Protocol tham khảo: [Bierling et al. 2025](https://www.nature.com/articles/s41597-025-04644-2).

## Mã hóa phản hồi

Notebook `analyses.ipynb` đã tải thành công, SHA-256:
`f3bc594a944f6276aba68bce0bb2ec5700fffaa7535e55d34687840efb001c9c`.
Chỉ đọc JSON/source cells, không thực thi notebook hoặc các output nhúng.

Cell index tính từ 0:

- Cell 6: tách healthy main đã inclusion, patients, excluded và retest.
- Cells 21/26: tính trung bình trực tiếp các descriptor nhị phân và so sánh
  với Keller đã đổi thang; không có khai báo `0=no, 1=yes` trực tiếp.

Đây là bằng chứng gián tiếp hỗ trợ cách diễn giải thông thường, chưa phải
codebook xác nhận chiều mã hóa. Trạng thái:
`INDIRECT_SUPPORT_NOT_EXPLICIT_ENCODING`. Không thay raw code thành nhãn.
Cần tài liệu export/LimeSurvey hoặc xác nhận tác giả ghi rõ `0/1`, cùng ý nghĩa
ô trống. Chưa liên hệ tác giả hoặc gửi dữ liệu ra ngoài.

## Kiểm tra cấu trúc nội bộ

Đếm trên **84 dòng metadata stimulus**, không phải 84 phân tử độc lập:

| Kiểm tra | Kết quả |
|---|---:|
| CAS đúng cú pháp/số kiểm tra | 84/84 |
| SMILES parse được bằng RDKit | 84/84 |
| RDKit phân loại achiral | 56 |
| RDKit tìm stereo đã khai báo | 17 |
| RDKit đánh dấu stereo chưa khai báo đầy đủ | 11 |
| Tên racemic nhưng SMILES chỉ định stereo | 1 |

Record `Citro`, dòng CSV 22: tên nguồn `(±)-beta-citronellol` nhưng SMILES
có `[C@@H]`. Giữ nguyên cả hai và đưa vào review về dạng mẫu; không biến
racemate thành một enantiomer được giả định.

Các mã có cờ stereo cần review: Decan, Octenol, Euca, Decalact, 3Hexa,
2Penta, Hexameth, Hydrohex, Triver, Menthiso, 4Carvo. Đây là kết quả kiểm tra
của RDKit, cần reviewer xem lại cả stereochemistry phụ thuộc vòng/cầu nối;
không mặc định mọi cờ đều là lỗi hóa học thực sự.

CAS checksum chỉ kiểm tra số, **không xác minh CAS/CID/SMILES cùng định danh**.
Mọi record còn yêu cầu đối chiếu database và dạng vật liệu nguồn. Chưa gửi
định danh tới PubChem trong bước này; không sửa CAS nào.

## Đối chiếu điều kiện đo

186 nhóm theo study/cohort/inclusion/set/molecule, bao phủ đúng 13.260 dòng:

| Kết quả tra metadata | Dòng phản hồi |
|---|---:|
| Cùng set và molcode, đề xuất đối chiếu | 9.613 |
| Anchor chung, đề xuất đối chiếu | 2.412 |
| Không tìm thấy cùng set | 35 |
| Patient giữ riêng để review | 480 |
| Test/retest giữ riêng để review | 720 |

35 dòng không có metadata cùng set đều là PEA ở set 2: 29 dòng inclusion=1,
6 dòng inclusion=0. Không chọn metadata set 1, 3 hoặc 9 để gán nồng độ thay.
Anchor và exact-set cũng chỉ là *proposed linkage*, chưa là context đã duyệt.
Cờ inclusion nguồn được giữ nguyên; không dùng các dòng excluded để train.

## Mapping cần duyệt riêng cho nguồn này

| Descriptor nguồn | Đề xuất trong 113 output | Trạng thái |
|---|---|---|
| sweet | sweet | Pending review |
| sour | sour | Pending review |
| fruit | fruity | Pending review |
| spices | spicy | Pending review |
| garlic | garlic | Pending review |
| fish | fishy | Pending review |
| burnt | burnt | Pending review |
| grass | grassy | Pending review |
| wood | woody | Pending review |
| flower | floral | Pending review |
| bakery | — | Source only |
| decayed | — | Source only |
| chemical | — | Source only |
| musky | — | Source only |
| sweaty | — | Source only |
| ammonia/urinous | — | Source only |

Duyệt mapping không đồng nghĩa duyệt phản hồi, concentration hoặc identity.
Không tự áp dụng quyết định mapping Keller cho CROWN. Không biến descriptor
không được hỏi thành ABSENT; không lấy intensity tổng của mùi làm intensity
riêng từng descriptor.

## Artifact và tái lập

Raw notebook:
`~/.scent-molecule-studio/raw/zenodo/bierling_2025/15657278-author-code`.

Audit private:
`~/.scent-molecule-studio/reviews/crown-2025-context-identity-audit-v1`:

- `identity_review.json`: raw record, cấu trúc RDKit, flags, CSV row provenance.
- `condition_review.json`: các source row đề xuất, số 0/1/missing theo điều kiện.
- `descriptor_review.json`: 16 quyết định đề xuất/source-only, chưa duyệt.
- `report.json`, `manifest.json`: kết quả, checksum nguồn/output/code.

Chạy bằng Python 3.10–3.12 với dependencies của training environment:

```bash
.venv-training/bin/python audit_crown_evidence.py \
  --raw-dir "$HOME/.scent-molecule-studio/raw/zenodo/bierling_2025/15657278" \
  --notebook-dir "$HOME/.scent-molecule-studio/raw/zenodo/bierling_2025/15657278-author-code" \
  --output "$HOME/.scent-molecule-studio/reviews/crown-2025-context-identity-audit-v1"
```

Đường dẫn output đã tồn tại sẽ bị từ chối; muốn audit mới phải dùng phiên bản
mới. Không có HTTP request trong công cụ audit. Artifact không chứa mã assessor
hoặc demographics; raw công khai vẫn giữ riêng ngoài Git.

Kiểm thử mới bao phủ checksum CAS không thành identity approval, racemate,
SMILES lỗi, stereo, conflict nội bộ, lookup PEA/anchor/retest, missing response,
immutable output, source corruption và không phát hành training target.
Đã chạy 49 tests liên quan thành công trên Python 3.12; chưa chạy matrix
Python 3.10/3.11 trong bước này.

## Điều kiện để đi tiếp

1. Codebook encoding hoặc xác nhận tác giả cho chiều 0/1/missing.
2. Review 10 mapping đề xuất; 6 descriptor còn lại giữ source-only.
3. Đối chiếu database/dạng mẫu và 35 dòng sai khác set; retest/context riêng.
4. Chọn consensus theo điều kiện, kiểm tra agreement và support độc lập.
5. Sau khi gate dữ liệu đạt mới tạo snapshot/split benchmark; chưa retrain.
