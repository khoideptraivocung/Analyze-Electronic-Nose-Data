# Mô Tả Tập Dữ Liệu Chất Lượng Không Khí (UCI Air Quality Dataset)

## 1. Tổng Quan Về Tập Dữ Liệu
Tập dữ liệu **Air Quality Dataset** (từ UCI Machine Learning Repository) chứa **9,358 bản ghi phản hồi trung bình theo giờ** từ mảng 5 cảm biến hóa học oxit kim loại (Metal Oxide - MOX) được tích hợp trong thiết bị Multisensor đo chất lượng không khí.

* **Thời gian thu thập:** Tháng 03/2004 đến Tháng 02/2005 (tròn 1 năm continuous deployment - đây là một trong những tập dữ liệu thực địa dài nhất hiện có về cảm biến không khí).
* **Vị trí đặt thiết bị:** Đặt thực địa ngang tầm đường đi tại một khu vực ô nhiễm không khí nặng thuộc một thành phố của Ý.
* **Dữ liệu chuẩn (Ground Truth - GT):** Nồng độ trung bình theo giờ của các chất khí: Carbon Monoxide ($\text{CO}$), Non-Metanic Hydrocarbons ($\text{NMHC}$), Benzene ($\text{C}_6\text{H}_6$), Total Nitrogen Oxides ($\text{NO}_x$), và Nitrogen Dioxide ($\text{NO}_2$) được đo đồng thời bởi một hệ thống phân tích tham chiếu chuẩn đã qua kiểm định (certified reference analyzer).

---

## 2. Trích Dẫn & Tài Liệu Tham Khảo (Citation Required)
> **De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008).**  
> *"On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario."*  
> **Sensors and Actuators B: Chemical**, 129(2), 750–757.

---

## 3. Bảng Giải Thích Các Trường Dữ Liệu (Data Schema)

| STT | Tên cột (Column) | Kiểu dữ liệu | Đơn vị / Khoảng giá trị | Mô tả chi tiết |
|---|---|---|---|---|
| 0 | `Date` | Chuỗi (String) | DD/MM/YYYY | Ngày ghi dữ liệu |
| 1 | `Time` | Chuỗi (String) | HH.MM.SS | Giờ ghi dữ liệu |
| 2 | `CO(GT)` | Số thực (Float) | $\text{mg/m}^3$ | Nồng độ $\text{CO}$ thực tế chuẩn (Reference Analyzer) |
| 3 | `PT08.S1(CO)` | Số thực (Float) | Đáp ứng tín hiệu / Điện trở | Phản hồi của Cảm biến Oxit Thiếc (Tin oxide) hướng đến khí $\text{CO}$ |
| 4 | `NMHC(GT)` | Số thực (Float) | $\mu\text{g/m}^3$ | Nồng độ Hydrocarbon phi-metan thực tế chuẩn (Reference Analyzer) |
| 5 | `C6H6(GT)` | Số thực (Float) | $\mu\text{g/m}^3$ | Nồng độ Benzene thực tế chuẩn (Reference Analyzer) |
| 6 | `PT08.S2(NMHC)` | Số thực (Float) | Đáp ứng tín hiệu / Điện trở | Phản hồi của Cảm biến Oxit Titin (Titania) hướng đến khí $\text{NMHC}$ |
| 7 | `NOx(GT)` | Số thực (Float) | $\text{ppb}$ | Nồng độ Tổng Nitrogen Oxides thực tế chuẩn (Reference Analyzer) |
| 8 | `PT08.S3(NOx)` | Số thực (Float) | Đáp ứng tín hiệu / Điện trở | Phản hồi của Cảm biến Oxit Vonfram (Tungsten oxide) hướng đến $\text{NO}_x$ |
| 9 | `NO2(GT)` | Số thực (Float) | $\mu\text{g/m}^3$ | Nồng độ Nitrogen Dioxide thực tế chuẩn (Reference Analyzer) |
| 10 | `PT08.S4(NO2)` | Số thực (Float) | Đáp ứng tín hiệu / Điện trở | Phản hồi của Cảm biến Oxit Vonfram (Tungsten oxide) hướng đến $\text{NO}_2$ |
| 11 | `PT08.S5(O3)` | Số thực (Float) | Đáp ứng tín hiệu / Điện trở | Phản hồi của Cảm biến Oxit Indium (Indium oxide) hướng đến Ozone $\text{O}_3$ |
| 12 | `T` | Số thực (Float) | $^\circ\text{C}$ | Nhiệt độ môi trường |
| 13 | `RH` | Số thực (Float) | $\%$ | Độ ẩm tương đối (Relative Humidity) |
| 14 | `AH` | Số thực (Float) | Số đo độ ẩm | Độ ẩm tuyệt đối (Absolute Humidity) |

---

## 4. Giải Thích Chi Tiết Về Giá Trị Mất Dữ Liệu (`-200`)

Trong tập dữ liệu này, **giá trị `-200` là Mã Đánh Dấu Dữ Liệu Bị Thiếu (Missing / Corrupted Data Sentinel Value)** do nhà nghiên cứu quy định. 

### Vì sao lại là `-200`?
Tất cả nồng độ khí thực tế, tín hiệu điện trở cảm biến, độ ẩm hay nhiệt độ ở thành phố Ý đều là **số dương** (hoặc nhiệt độ không bao giờ xuống tới $-200^\circ\text{C}$). Do đó con số âm bất thường `-200` được chọn để đánh dấu rõ các điểm dữ liệu bị khuyết mà không sợ trùng lặp với số liệu đo thật.

### Thống kê & Nguyên nhân bị mất dữ liệu (`-200`):

| Cột dữ liệu | Số lượng bị `-200` | Tỷ lệ bị khuyết | Nguyên nhân thực địa |
|---|---|---|---|
| `Date`, `Time` | 0 dòng | 0.00% | Đầy đủ trục thời gian |
| `CO(GT)` | 1,683 dòng | 17.99% | Thiết bị chuẩn bị tạm ngừng / hiệu chuẩn định kỳ / lỗi ống lấy mẫu |
| `PT08.S1(CO)` | 366 dòng | 3.91% | Thiết bị mảng cảm biến bị mất nguồn hoặc ngắt kết nối tạm thời |
| `NMHC(GT)` | 8,443 dòng | **90.23%** | **Máy đo NMHC chuẩn bị hỏng/ngừng hoạt động hoàn toàn** sau tháng 4/2004 |
| `C6H6(GT)` | 366 dòng | 3.91% | Thiết bị chuẩn thiếu dữ liệu |
| `PT08.S2(NMHC)` | 366 dòng | 3.91% | Cảm biến mất tín hiệu tạm thời |
| `NOx(GT)` | 1,639 dòng | 17.52% | Thiết bị chuẩn bị tạm ngừng / hiệu chuẩn |
| `PT08.S3(NOx)` | 366 dòng | 3.91% | Cảm biến mất tín hiệu tạm thời |
| `NO2(GT)` | 1,642 dòng | 17.55% | Thiết bị chuẩn bị tạm ngừng / hiệu chuẩn |
| `PT08.S4(NO2)` | 366 dòng | 3.91% | Cảm biến mất tín hiệu tạm thời |
| `PT08.S5(O3)` | 366 dòng | 3.91% | Cảm biến mất tín hiệu tạm thời |
| `T`, `RH`, `AH` | 366 dòng | 3.91% | Bộ đo môi trường bị ngắt tín hiệu cùng 366 dòng của mảng cảm biến |

---

## 5. Các Thách Thức Phân Tích Dữ Liệu Chuỗi Thời Gian
1. **Giao thoa nhạy (Cross-sensitivities):** Các cảm biến oxit kim loại không chỉ phản hồi với khí mục tiêu mà còn phản hồi chéo với các khí khác và phụ thuộc mạnh vào nhiệt độ ($T$) cũng như độ ẩm ($RH, AH$).
2. **Hiện tượng trôi cảm biến & trôi khái niệm (Sensor Drift & Concept Drift):** Đáp ứng của cảm biến thay đổi và bị suy giảm dần theo thời gian trong suốt 1 năm hoạt động liên tục ngoài thực địa.
3. **Mất liên tục thời gian:** Các khoảng giá trị `-200` cần được chuyển về `NaN` và xử lý nối chuỗi thời gian (interpolation / resample) trước khi vẽ đồ thị.
