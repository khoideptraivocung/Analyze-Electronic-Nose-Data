# Mô Tả Tập Dữ Liệu Phát Hiện Khí Nâng Cao (Advanced Gas Detection Dataset)

## 1. Tổng Quan Về Tập Dữ Liệu
Tập dữ liệu **Advanced Gas Detection and Classification using MQ Series Sensors** được xây dựng nhằm phục vụ nghiên cứu phát hiện và phân loại các loại khí ô nhiễm/độc hại bằng mảng cảm biến chuỗi MQ kết hợp với các mô hình Học máy (Machine Learning) và Học sâu (Deep Learning).

* **Số lượng mẫu:** **100,422 mẫu (instances)** theo chuỗi thời gian.
* **Thời gian thu thập:** 10 ngày (từ ngày 22/02/2023 đến ngày 03/03/2023).
* **Phần cứng & Mô phỏng:** Hệ thống mảng 6 cảm biến thuộc chuỗi MQ (bao gồm **MQ-135, MQ-5, MQ-6**) kết nối với vi điều khiển **Arduino UNO**, được mô phỏng trên phần mềm **Proteus**.
* **Truyền dữ liệu:** Dữ liệu từ vi điều khiển được truyền đến giao diện **LabVIEW GUI** để trực quan hóa và lưu trữ thành định dạng `.csv`.

---

## 2. Các Loại Khí Mục Tiêu & Mảng Cảm Biến

Hệ thống theo dõi và phát hiện **6 loại khí chính**:
1. **Ammonia ($\text{NH}_3$)** - Khí amoniac
2. **Carbon Dioxide ($\text{CO}_2$)** - Khí cacbonic
3. **Benzene ($\text{C}_6\text{H}_6$)** - Khí benzen
4. **Natural Gas ($\text{CH}_4$)** - Khí tự nhiên (Metan)
5. **Carbon Monoxide ($\text{CO}$)** - Khí cacbon monoxit
6. **Liquefied Petroleum Gas (LPG)** - Khí hóa lỏng (Gas đun nấu)

Mảng cảm biến gồm 6 kênh đáp ứng tín hiệu tương ứng từ `Gas1` đến `Gas6` và giá trị nồng độ quy đổi tương ứng tính bằng **PPM (parts per million)** từ `Gas1 PPM` đến `Gas6 PPM`.

---

## 3. Cấu Trúc Dữ Liệu (Data Schema)

Tập dữ liệu gồm **15 cột (variables)**:

| STT | Tên Cột | Kiểu Dữ Liệu | Đơn Vị / Thang Đo | Mô Tả Chi Tiết |
|---|---|---|---|---|
| 0 | `Date` | String | DD/MM/YYYY | Ngày thu thập dữ liệu (22/02/2023 – 03/03/2023) |
| 1 | `Time(sec)` | Float | Giây (sec) | Dấu thời gian ghi mẫu (Timestamp) |
| 2 | `Gas1` | Float | Giá trị thô / ADC | Tín hiệu điện áp thô từ Cảm biến 1 |
| 3 | `Gas2` | Float | Giá trị thô / ADC | Tín hiệu điện áp thô từ Cảm biến 2 |
| 4 | `Gas3` | Float | Giá trị thô / ADC | Tín hiệu điện áp thô từ Cảm biến 3 |
| 5 | `Gas4` | Float | Giá trị thô / ADC | Tín hiệu điện áp thô từ Cảm biến 4 |
| 6 | `Gas5` | Float | Giá trị thô / ADC | Tín hiệu điện áp thô từ Cảm biến 5 |
| 7 | `Gas6` | Float | Giá trị thô / ADC | Tín hiệu điện áp thô từ Cảm biến 6 |
| 8 | `Gas1 PPM` | Float | PPM | Nồng độ khí quy đổi của Cảm biến 1 |
| 9 | `Gas2 PPM` | Float | PPM | Nồng độ khí quy đổi của Cảm biến 2 |
| 10 | `Gas3 PPM` | Float | PPM | Nồng độ khí quy đổi của Cảm biến 3 |
| 11 | `Gas4 PPM` | Float | PPM | Nồng độ khí quy đổi của Cảm biến 4 |
| 12 | `Gas5 PPM` | Float | PPM | Nồng độ khí quy đổi của Cảm biến 5 |
| 13 | `Gas6 PPM` | Float | PPM | Nồng độ khí quy đổi của Cảm biến 6 |
| 14 | `Class` | Integer | 0, 1, 2, 3, 4, 5, 6 | **Nhãn phân loại (Target Class)** chỉ định trạng thái có/không hoặc loại khí có mặt |

---

## 4. Phân Bố 7 Lớp Khí (Class Distribution)

| Nhãn (`Class`) | Số Lượng Mẫu | Tỷ Lệ (%) | Trạng Thái / Loại Khí |
|---|---|---|---|
| `0` | 9,136 | 9.10% | Khí sạch (Clean Air / Không có khí độc) |
| `1` | 7,585 | 7.55% | Trạng thái khí loại 1 |
| `2` | 17,309 | 17.24% | Trạng thái khí loại 2 |
| `3` | 22,765 | 22.67% | Trạng thái khí loại 3 (Chiếm tỷ lệ cao nhất) |
| `4` | 22,186 | 22.09% | Trạng thái khí loại 4 |
| `5` | 11,027 | 10.98% | Trạng thái khí loại 5 |
| `6` | 10,414 | 10.37% | Trạng thái khí loại 6 |

---

## 5. Đặc Điểm Chất Lượng & Tiền Xử Lý Dữ Liệu
* **Dữ liệu khuyết thiếu (Missing Values):** 0 mẫu thiếu (100,422 mẫu đều đầy đủ thông tin).
* **Biến động thời gian:** Thu thập liên tục trong 10 ngày với nhịp lấy mẫu theo giây.
* **Xử lý Outlier & Scaling:** Dữ liệu đã qua bước xử lý lọc nhiễu ngoại lệ (outlier removal) và chuẩn hóa thang đo (scaling) từ Proteus/LabVIEW để phục vụ các mô hình Học máy và DL.
