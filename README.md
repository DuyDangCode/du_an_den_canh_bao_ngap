# Đèn Nội Thất Thông Minh Dự Báo và Cảnh Báo Ngập Lụt Đô Thị (Bản Mô Phỏng Phần Mềm 3D & Machine Learning)

> **"360 ngày là đèn trang trí sang trọng – Ngày mưa bão là thiết bị cứu hộ chủ động."**  
> *(Đồ án Tiền Tốt Nghiệp – Mô phỏng Không gian 3D, Cảm biến Ảo & Mô hình Machine Learning Học Thích Nghi)*

---

## 1. Giới thiệu & Phạm vi Dự án (Project Scope)

Dự án **Đèn Nội Thất Thông Minh Dự Báo và Cảnh Báo Ngập Lụt Đô Thị** mang đến giải pháp tích hợp hệ thống cảm biến "tàng hình", bộ não Trí tuệ nhân tạo (Machine Learning) dự báo ngập sớm và cơ chế cứu hộ khẩn cấp vào bên trong một chiếc đèn trang trí nội thất cao cấp (một module độc lập của Nhà thông minh).

### 🎯 Phạm vi triển khai hiện tại (100% Software Simulation)
Trong giai đoạn hiện tại, **toàn bộ dự án được triển khai hoàn toàn trên phần mềm (Web Application + Python ML Backend)** và **không kết nối với thiết bị phần cứng thật**. Mọi thành phần vật lý (ngôi nhà, chiếc đèn nội thất, mạng cảm biến, mực nước ngập và mô hình AI dự báo) đều được số hóa và mô phỏng trực quan qua **2 trang chức năng chính**:

1. **Trang Demo Mô phỏng 3D Chính (Main 3D Simulation Page)**:
   - Tải trực tiếp mô hình 3D ngôi nhà từ [`resources/house.stl`](file:///home/thanhduy/Projects/du_an_den_canh_bao_ngap/resources/house.stl) lên không gian 3D (Three.js) kèm **hiệu ứng xoay 360°** (Orbit/Auto-rotate).
   - Hiển thị trực quan **mô hình 3D chiếc đèn nội thất thông minh** (với dải LED 4 màu đóng vai trò thước đo mực nước theo cm, đèn trắng rọi sàn, đầu đèn tháo rời nam châm) và **mặt nước 3D dâng lên** theo độ sâu ngập.
   - Hiển thị trạng thái loa cảnh báo dưới dạng **hiệu ứng sóng âm & thông điệp trực quan trên màn hình (Visual-only)** (không phát âm thanh thật qua trình duyệt).
   - Cung cấp **Bảng điều khiển mô phỏng môi trường & cảm biến** (thay cho tín hiệu cảm biến phần cứng) để người dùng/giảng viên tự do điều chỉnh Lượng mưa, Triều cường, Độ ẩm, Áp suất, Mực nước cống, Tốc độ nước dâng, Mực nước sàn, Thời gian nước đọng (test lọc chống báo giả 30s), trạng thái Điện/Pin dự phòng và các nút Kịch bản mẫu 1-Click.

2. **Trang Kết quả Dự báo Machine Learning & Tinh chỉnh Ngưỡng (ML Forecast, Online Learning & Thresholds Page)**:
   - **Mô hình Machine Learning nhẹ có khả năng Học Trực tuyến / Học Tăng cường (Online / Incremental Learning)**:
     - Được huấn luyện nền (Pre-trained) trên **bảng dữ liệu mô phỏng lịch sử ngập tại TP.HCM** (kết hợp mưa lớn và triều cường).
     - Nhận đầu vào từ Trang Demo để tính toán tức thì (`< 5ms`): **Xác suất ngập (%)**, **Độ sâu ngập dự kiến trong 30–60 phút tới (cm)**, mức độ ảnh hưởng của từng yếu tố, bản đồ ngập theo khu vực trong nhà và **Danh sách việc cần làm khẩn cấp (Emergency Checklist)**.
   - **Tính năng "Dạy AI Học Thích Nghi" (Online Learning / Reward Feedback)**:
     - Mỗi ngôi nhà có độ cao cốt nền khác nhau. Trang Kết quả cho phép nhập **mức ngập thực tế sau cơn mưa** để mô hình cập nhật trọng số tức thì (`partial_fit` / Online Gradient Descent), giúp Hội đồng chấm thi kiểm chứng trực tiếp khả năng "tự học thói quen từng nhà" của AI qua các cơn mưa.
   - **Tinh chỉnh các ngưỡng cấu hình (Threshold Configuration)**: Độ cao cốt nền nhà, chiều cao thân đèn (60–100 cm), ngưỡng kích hoạt Cấp 1–2–3 và thời gian lọc chống báo động giả (mặc định 30 giây).

---

## 2. Bốn Cấp độ Biểu diễn Ánh sáng & Cảnh báo (Mô phỏng 3D + Visual)

| Cấp độ | Trạng thái | Điều kiện Kích hoạt (Giả lập / Dự báo ML) | Biểu diễn Đèn 3D | Hiển thị Loa (Visual-Only) & Hành động |
| :--- | :--- | :--- | :--- | :--- |
| **Cấp 0** | **Ngày bình thường** *(Normal Mode)* | Thời tiết bình thường, cống và sàn khô ráo | **Vàng ấm (Warm White)** dịu mắt, làm đèn trang trí nội thất | Sóng âm tắt – Trạng thái: *Đèn trang trí nội thất* |
| **Cấp 1** | **Dự báo sớm** *(Early Warning)* | Trước ~60 phút (Nguy cơ mưa lớn + triều cường từ ML & ngưỡng Cấp 1) | **Xanh dương dịu** hiệu ứng gợn sóng nhẹ như mặt nước | Hiệu ứng sóng âm nhẹ: *"Chuông gió – Cảnh báo mưa lớn & triều cường chiều tối nay"* |
| **Cấp 2** | **Chuẩn bị ứng phó** *(Pre-Flood Alert)* | Trước 15–30 phút (Nước cống đang dâng, ML dự báo sắp ngập $H\text{ cm}$) | **Cam**, cột sáng dâng cao đúng bằng mức nước dự kiến ngập (VD: 15 cm) | Hiệu ứng sóng âm giọng nói: *"Nhắc chủ nhà lắp ván chắn cửa, kê cao đồ đạc"* |
| **Cấp 3** | **Khẩn cấp** *(Emergency)* | Nước chạm sàn $\ge 1\text{ cm}$ liên tục $> 30\text{s}$ kèm cống đầy hoặc trời mưa | **Chớp Đỏ** + **Đèn trắng rọi thẳng xuống sàn** chống vấp ngã | Hiệu ứng sóng âm khẩn cấp: *"Đang hú còi lớn đánh thức cả nhà"* (Có nút **Tắt còi / Mute**) |

---

## 3. Các Cơ chế Cốt lõi Được Mô phỏng trên Phần mềm

- **Mô hình Machine Learning Học Thích Nghi (Online Learning)**: Siêu nhẹ, chạy hoàn toàn cục bộ (không phụ thuộc Cloud/API bên ngoài), dự báo tức thì số $\text{cm}$ ngập và tự điều chỉnh trọng số sau mỗi lần nhận phản hồi thực tế của ngôi nhà.
- **Chống báo động giả (Lau nhà / Đổ nước sinh hoạt)**: Nếu kéo thanh trượt có nước trên sàn nhà nhưng thời gian nước đọng $< 30\text{ giây}$ HOẶC dưới cống khô ráo & trời không mưa, hệ thống nhận diện là nước sinh hoạt và không kích hoạt báo động đỏ Cấp 3.
- **Mô phỏng Mất điện / Chạy Pin dự phòng (8–12h) & Offline Mode**: Cho phép bật công tắc giả lập cắt điện lưới và mất mạng Wi-Fi để kiểm chứng đèn vẫn tự sáng bằng pin tích hợp và dự báo độc lập tại chỗ.
- **Mô phỏng Tháo rời Đầu đèn Nam châm**: Cho phép bật/tắt trạng thái nhấc rời phần đầu đèn khỏi thân đèn để làm đèn pin cầm tay cứu hộ.

---

## 4. Tài liệu Dự án (`docs/`)

Toàn bộ tài liệu thiết kế và đặc tả của dự án được lưu trữ trong thư mục [`docs/`](file:///home/thanhduy/Projects/du_an_den_canh_bao_ngap/docs):

- 📄 [**Báo cáo Đề tài Gốc (`docs/proposal.pdf`)**](file:///home/thanhduy/Projects/du_an_den_canh_bao_ngap/docs/proposal.pdf): Tài liệu thuyết minh ý tưởng gốc của đồ án tiền tốt nghiệp.
- 🏗️ [**Cấu trúc Sản phẩm & Cấu trúc Thư mục (`docs/product-and-folder-structure.md`)**](file:///home/thanhduy/Projects/du_an_den_canh_bao_ngap/docs/product-and-folder-structure.md): Kiến trúc phần mềm mô phỏng (Trang Demo 3D Three.js, Mô hình ML Học Trực tuyến, Trang Kết quả & Cấu hình ngưỡng) và cấu trúc cây thư mục dự án.
- 📋 [**Đặc tả Yêu cầu Chức năng (`docs/functional-requirements.md`)**](file:///home/thanhduy/Projects/du_an_den_canh_bao_ngap/docs/functional-requirements.md): Chi tiết các yêu cầu chức năng (FR) và phi chức năng (NFR) theo phạm vi 100% phần mềm mô phỏng.
- 🏠 [**Mô hình 3D Ngôi nhà (`resources/house.stl`)**](file:///home/thanhduy/Projects/du_an_den_canh_bao_ngap/resources/house.stl): Tệp mô hình 3D ngôi nhà được tải trực tiếp lên Trang Demo 3D.
