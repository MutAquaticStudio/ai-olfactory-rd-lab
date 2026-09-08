# Nguồn panel bổ sung trước retrain

Trạng thái CROWN mới nhất: [encoding, mapping và training preflight](CROWN_TRAINING_PREFLIGHT.md).

Ngày kiểm tra: 08/09/2026. Phạm vi: tìm evidence có protocol rõ và quyền sử dụng
phù hợp, không chạy training, không mua dữ liệu hoặc liên hệ tác giả.

## Quyết định nguồn

| Nguồn | Quyết định | Căn cứ |
|---|---|---|
| Bierling/CROWN 2025 | Đã tải và đưa vào staging review private | Protocol hỏi yes/no cho 16 descriptor; Zenodo v3 khai báo CC BY 4.0 |
| Dravnieks 1985 | Chưa tải | ASTM công bố hạn chế sử dụng nội dung bằng AI; cần quyền phù hợp trước khi nhập |
| OdorNet | Không dùng để thay assessed negatives của panel | Là dữ liệu tích hợp/taxonomy 12 nhóm; README nêu một số mã 0 dựa trên giả định mô hình, không chứng minh đã được assessor đánh giá âm |

Nguồn đối chiếu: [Bierling et al., Methods](https://www.nature.com/articles/s41597-025-04644-2),
[Zenodo v3](https://zenodo.org/records/15657278),
[metadata giấy phép Zenodo](https://zenodo.org/api/records/15657278),
[ASTM](https://store.astm.org/ds61-eb.html),
[OdorNet: Data format](https://github.com/NKU-DOIE/OdorNet#data-format).

## CROWN: dữ liệu đã kiểm tra

Pin record `15657278`, DOI `10.5281/zenodo.15657278`, phiên bản v3. Chỉ tải
`data.csv`, `odors.csv` và `variables-dictionary.xlsx`; kiểm tra MD5/size từ
Zenodo, lưu thêm SHA-256 và metadata giấy phép. Attribution: Bierling và cộng
sự (2025). Không thực thi notebook của tác giả.

Kiểm tra trực tiếp các file đã tải:

- 13.260 dòng participant–odor–occasion; 212.160 vị trí phản hồi descriptor.
- 165.386 mã `0`, 40.292 mã `1`, 6.482 ô thiếu; giữ nguyên ba loại.
- 84 dòng metadata stimulus, 74 mã; chỉ 73 mã xuất hiện trong bảng phản hồi.
  `4Isoprop` chỉ có ở metadata. Không tự tạo dòng đánh giá cho mã này.
- Các nhóm được giữ riêng: home, lab, patient, test và retest; 870 dòng bị loại
  theo cờ inclusion nguồn vẫn được giữ với trạng thái excluded.
- Retest có 60 participant codes, mỗi người được ghi nhận ở hai lần đo trên
  6 odorant. Không tính hai lần đo thành 120 người.
- Có mã molecule được dùng ở nhiều nhóm và nồng độ khác nhau, ví dụ PEA.
  Chưa join bằng `molcode` đơn lẻ để tránh gán nhầm điều kiện.

Từ điển biến xác nhận yes/no tại sheet `Data Record 1 - main data`, hàng
61–76. Có khác biệt tên: `fruity` trong dictionary nhưng `fruit` trong CSV;
dictionary viết `ammonia/urinuos`, CSV dùng `ammonia/urinous`. Mọi alias phải
được ghi trong mapping nguồn, không fuzzy-map ngầm.

**Chiều mã hóa 0/1 còn cần đối chiếu:** dictionary mô tả yes/no nhưng không
ghi trực tiếp số tương ứng; hai lần thử đọc notebook chính thức bị timeout.
Staging vì vậy lưu `RECORDED_ZERO`, `RECORDED_ONE`, `MISSING`; chưa chuyển thành
PRESENT/ABSENT. Các `presence_state` training vẫn là UNASSESSED, intensity null.

Cập nhật bước tiếp: đã tải được notebook đúng checksum và hoàn tất kiểm tra
mã tĩnh. Không thấy khai báo trực tiếp chiều mã hóa; xem
[audit protocol, identity và context](CROWN_PROTOCOL_IDENTITY_AUDIT.md).
Staging v1 được giữ nguyên để bảo toàn lịch sử, không sửa nội dung artifact cũ.

## Artifacts và giới hạn

Raw: `~/.scent-molecule-studio/raw/zenodo/bierling_2025/15657278`.
Review: `~/.scent-molecule-studio/reviews/crown-2025-source-audit-v1`.

`source_responses.parquet` chỉ giữ trường định danh pseudonymous, cohort,
occasion và phản hồi cảm quan cần review. Không đưa demographics, bảng điểm sức
khỏe, personality hoặc free text vào bảng này. Raw công khai vẫn được giữ riêng,
không commit hoặc gửi ra ngoài. `source_stimuli.parquet` giữ raw SMILES và kết quả
RDKit source-level; không xác nhận lô mẫu, CAS/CID hoặc stereo thực nghiệm.

Đây là staging riêng, **chưa đăng ký thành training snapshot trong Data
Foundation**, chưa cộng số lượng vào coverage Keller và chưa đổi model.

Mười mapping rộng/literal được đề xuất, chưa duyệt: sweet, sour, fruity, spicy,
garlic, fishy, burnt, grassy, woody, floral. MUSKY vẫn giữ riêng; không có câu
hỏi trực tiếp về jasmine hoặc citrus trong 16 descriptor.

## Bước tiếp theo

1. Xác nhận chiều mã hóa 0/1 bằng code hoặc tài liệu encoding của nguồn.
2. Duyệt mapping riêng cho tiếng Đức/Anh và protocol yes/no của CROWN.
3. Xác minh định danh và join đúng cohort/set/concentration; giữ retest riêng.
4. Tính agreement và consensus theo điều kiện rồi đếm support theo phân tử.
5. Chỉ sau data gate mới phát hành snapshot và split benchmark mới.

73 mã có phản hồi không tự bảo đảm 50 positive và 50 negative **phân tử độc
lập, không giao nhau** cho một nhãn, càng không bảo đảm support trong calibration
partition. Nguồn này bổ sung evidence có giá trị, không tự mở gate retrain.
