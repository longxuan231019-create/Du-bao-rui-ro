# 🕵️ Hệ thống Phát hiện Giao dịch Gian lận (Fraud Detection App)

Ứng dụng web Streamlit phân tích và dự báo rủi ro tín dụng/gian lận, được chuyển đổi tự động từ Notebook Machine Learning. Ứng dụng hỗ trợ đọc dữ liệu, trực quan hóa đặc trưng, huấn luyện các thuật toán phân loại và triển khai dự báo trên dữ liệu thực tế.

## Tính năng chính (Các Tab)
- **📊 Tổng quan dữ liệu:** Xem kích thước, dữ liệu thô và thống kê mô tả của các biến.
- **📈 Trực quan hóa:** Phân phối của biến mục tiêu và histogram của các biến đầu vào, phân tách theo nhãn (Bình thường / Gian lận).
- **🎯 Kết quả kiểm định:** Báo cáo Classification Report và Confusion Matrix ngay sau khi huấn luyện.
- **🔮 Sử dụng mô hình:** Cung cấp tính năng dự báo đơn lẻ (nhập tay) và dự báo hàng loạt bằng file Excel/CSV.

## Mô hình áp dụng
Theo mặc định của Notebook, ứng dụng hỗ trợ 3 mô hình phân loại:
1. **Random Forest** (Mặc định)
2. **Decision Tree**
3. **Logistic Regression**

*(Các siêu tham số như `n_estimators`, `max_depth`, `max_iter` có thể tùy chỉnh trực tiếp tại thanh điều khiển bên trái).*

## Cấu trúc dữ liệu đầu vào
Tệp tin dữ liệu (CSV/Excel) tải lên phải chứa tối thiểu các cột sau:
- Các cột đặc trưng: `X_1`, `X_2`, `X_3`, `X_4`, `X_5`, `X_6`, `X_7`, `X_8`, `X_9`, `X_10`, `X_11`, `X_12`, `X_13`, `X_14` (Kiểu số - Float/Int)
- Cột mục tiêu (Chỉ dùng khi huấn luyện): `default` (Kiểu số 0 hoặc 1).

## Hướng dẫn cài đặt và chạy ứng dụng

1. **Cài đặt thư viện yêu cầu:**
   Mở terminal / command prompt và chạy lệnh sau để cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
