# Analyze Electronic Nose Data

Dự án phân tích và trực quan hóa dữ liệu từ cảm biến mũi điện tử (Electronic Nose - E-Nose) sử dụng tập dữ liệu từ Kaggle.

## Mục lục
- [Giới thiệu](#giới thiệu)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Hướng dẫn cài đặt và chạy thử](#hướng-dẫn-cài-đặt-và-chạy-thử)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Tạo môi trường ảo (Virtual Environment)](#2-tạo-môi-trường-ảo-virtual-environment)
  - [3. Cài đặt các thư viện cần thiết](#3-cài-đặt-các-thư-viện-cần-thiết)
  - [4. Chạy các tập lệnh phân tích](#4-chạy-các-tập-lệnh-phân-tích)
- [Danh sách các thư viện chính](#danh-sách-các-thư-viện-chính)

## Giới thiệu
Dự án này thực hiện phân tích thống kê và vẽ các biểu đồ trực quan hóa (như phân bố lớp mẫu, chuỗi thời gian của các cảm biến, phân cụm PCA, heatmap tương quan, v.v.) từ các dữ liệu thu thập bởi mảng cảm biến khí ga. 

Dữ liệu được tự động tải từ Kaggle thông qua thư viện `kagglehub`, giúp tối ưu dung lượng của repository và luôn đảm bảo dữ liệu chạy là mới nhất.

## Cấu trúc thư mục
```text
├── Gas_Classification_Dataset_tunnel/  # Phân tích dữ liệu phân loại khí trong wind tunnel
│   ├── analyze_gas_wind.py            # Script thực hiện PCA, vẽ Heatmap và phân bố mẫu
│   ├── plot_fingerprint_bar.py        # Vẽ dấu vân tay hóa chất của cảm biến
│   └── ...                            # Các biểu đồ được tạo ra
├── dynamic_gas/                       # Phân tích dữ liệu cảm biến khí động học (dynamic gas)
│   ├── analyze_dynamic_gas.py         # Script phân tích nồng độ khí Ethylene, CO, Methane
│   └── ...                            # Các biểu đồ trực quan hóa dữ liệu động
├── paper/                             # Chứa tài liệu nghiên cứu tham khảo dạng PDF
├── analyze_data.py                    # Script phân tích tổng quan dữ liệu mẫu
├── plot_data.py                       # Script vẽ biểu đồ cảm biến cơ bản
├── requirements.txt                   # Danh sách thư viện phụ thuộc
└── README.md                          # Tài liệu hướng dẫn sử dụng
```

## Hướng dẫn cài đặt và chạy thử

### 1. Clone Repository
Mở Terminal/Command Prompt và chạy lệnh sau để clone repository về máy của bạn:
```bash
git clone https://github.com/khoideptraivocung/Analyze-Electronic-Nose-Data.git
cd Analyze-Electronic-Nose-Data
```

### 2. Tạo môi trường ảo (Virtual Environment)
Để tránh xung đột thư viện giữa các dự án khác nhau, bạn nên tạo môi trường ảo Python:

- **Trên Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **Trên macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Cài đặt các thư viện cần thiết
Sử dụng `pip` để cài đặt các thư viện phụ thuộc từ file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Chạy các tập lệnh phân tích
Sau khi cài đặt xong thư viện, bạn có thể chạy bất kỳ script nào để tải tập dữ liệu tự động từ Kaggle và tạo biểu đồ phân tích:

- **Phân tích tổng quan dữ liệu mẫu:**
  ```bash
  python analyze_data.py
  ```

- **Phân tích dữ liệu phân loại khí ga (Wind Tunnel):**
  ```bash
  cd Gas_Classification_Dataset_tunnel
  python analyze_gas_wind.py
  ```

- **Phân tích dữ liệu khí động học động:**
  ```bash
  cd ../dynamic_gas
  python analyze_dynamic_gas.py
  ```

## Danh sách các thư viện chính
- `kagglehub`: Tải tự động và lưu trữ cache dataset từ Kaggle.
- `pandas`: Xử lý và thao tác với cấu trúc dữ liệu bảng dữ liệu cảm biến.
- `matplotlib` & `seaborn`: Thiết kế và xuất các biểu đồ trực quan chất lượng cao.
- `scikit-learn`: Thực hiện tiền xử lý dữ liệu và thuật toán giảm chiều dữ liệu PCA (Phân tích thành phần chính).
