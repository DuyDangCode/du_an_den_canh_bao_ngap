# Đặc Tả Yêu Cầu Chức Năng và Phi Chức Năng (Functional & Non-Functional Requirements)

> **Tên đề tài**: Thiết kế đèn nội thất thông minh dự báo và cảnh báo ngập lụt đô thị  
> **Ý tưởng cốt lõi**: *"360 ngày là đèn trang trí sang trọng – Ngày mưa bão là thiết bị cứu hộ chủ động"*  
> **Phạm vi triển khai hiện tại (Scope)**: **100% Phần mềm Mô phỏng (Software-only Web Application)** – Không kết nối phần cứng thật. Toàn bộ ngôi nhà ([`resources/house.stl`](file:///home/thanhduy/Projects/du_an_den_canh_bao_ngap/resources/house.stl)), chiếc đèn nội thất, tín hiệu cảm biến, bộ lọc chống báo giả và **Mô hình Machine Learning Học Thích Nghi / Học Tăng cường Trực tuyến (Online / Incremental Learning)** được triển khai qua **Trang Demo 3D Chính** và **Trang Kết quả Dự báo & Cấu hình Ngưỡng**.

---

## 1. Đối tượng Sử dụng và Tác nhân Hệ thống (Actors)

| Tác nhân (Actor) | Vai trò & Mục tiêu sử dụng trên Phần mềm |
| :--- | :--- |
| **Giảng viên / Hội đồng chấm thi & Người dùng** | - Quan sát mô hình 3D ngôi nhà (`house.stl`) xoay 360° cùng chiếc đèn thông minh 3D và mực nước ngập trên **Trang Demo**.<br>- Kéo các thanh trượt điều kiện môi trường & cảm biến ảo (hoặc chọn kịch bản mẫu 1-Click) để kiểm chứng phản ứng tức thì của đèn và bộ lọc chống báo giả 30 giây.<br>- Xem báo cáo dự báo từ **Mô hình Machine Learning**, nhập mức ngập thực tế sau cơn mưa để **dạy AI học thích nghi** và tùy chỉnh các ngưỡng cấu hình tại **Trang Kết quả**. |
| **Bộ não Machine Learning Học Trực tuyến (`ml_engine`)** | - Huấn luyện nền trên tập dữ liệu lịch sử ngập TP.HCM.<br>- Nhận bộ thông số đầu vào (Thời tiết, Tín hiệu cảm biến giả lập, Độ cao cốt nền nhà) để dự báo tức thì (`< 5ms`) xác suất ngập ($\%$) và độ sâu ngập dự kiến ($\text{cm}$) trong 30–60 phút tới.<br>- Tự động cập nhật trọng số (`partial_fit` / Online Gradient Descent) khi nhận phản hồi mức ngập thực tế của ngôi nhà. |

---

## 2. Yêu cầu Chức năng (Functional Requirements - FR)

### 2.1. Phân hệ 1: Không gian 3D & Biểu diễn Đèn Nội thất trên Trang Demo (`FR-3D`)

| Mã YC | Tên chức năng | Mô tả chi tiết yêu cầu | Mức ưu tiên |
| :--- | :--- | :--- | :---: |
| **FR-3D-01** | **Tải và xoay 360° mô hình 3D ngôi nhà (`house.stl`)** | Trang Demo chính phải tải trực tiếp file [`resources/house.stl`](file:///home/thanhduy/Projects/du_an_den_canh_bao_ngap/resources/house.stl) lên không gian 3D (Three.js), căn chỉnh tỷ lệ/ánh sáng hài hòa và hỗ trợ **hiệu ứng tự động xoay (Auto-rotate)** cũng như cho phép người dùng dùng chuột/cảm ứng xoay 360°, phóng to, thu nhỏ góc nhìn. | Bắt buộc |
| **FR-3D-02** | **Mô phỏng 3D chiếc đèn nội thất và 4 cấp độ màu sắc** | Dựng mô hình 3D chiếc đèn nội thất trung tâm (tương ứng chiều cao 60–100 cm, chất liệu gỗ & kính tán sáng mờ) ngay trong không gian 3D với khả năng đổi màu theo 4 cấp độ:<br>1. **Cấp 0 – Ngày bình thường**: Sáng màu **Vàng ấm (Warm White)**.<br>2. **Cấp 1 – Dự báo sớm (Trước ~60 phút)**: Chuyển màu **Xanh dương dịu** kèm hiệu ứng chuyển động gợn sóng nhẹ như mặt nước.<br>3. **Cấp 2 – Chuẩn bị ứng phó (Trước 15–30 phút)**: Chuyển màu **Cam**.<br>4. **Cấp 3 – Khẩn cấp (Nước tràn vào sàn nhà)**: Chớp **Đỏ** cảnh báo nguy hiểm. | Bắt buộc |
| **FR-3D-03** | **Hiển thị "Thước đo mực nước" bằng cột sáng 3D trên thân đèn** | Dải sáng LED trên thân đèn 3D phải sáng cao dần từ chân đèn lên tương ứng với **tỷ lệ chiều cao mực nước dự kiến ngập ($\text{cm}$) trên tổng chiều cao thân đèn ($60\text{–}100\text{ cm}$)** (ví dụ: mô hình ML dự báo ngập $15\text{ cm}$ trên thân đèn cao $75\text{ cm}$ thì cột sáng dâng cao đúng $20\%$ chiều cao thân đèn và hiển thị nhãn `15 cm`). | Bắt buộc |
| **FR-3D-04** | **Mô phỏng Đèn trắng rọi sàn (Floor Downlight) & Mặt nước 3D** | - Khi ở trạng thái **Khẩn cấp (Cấp 3)**, mô hình đèn 3D phải tự động bật **luồng sáng trắng rọi thẳng xuống mặt sàn**.<br>- Đồng thời, không gian 3D hiển thị **mặt nước bán trong suốt dâng lên** quanh/trong mô hình `house.stl` tương ứng với mực nước ngập. | Bắt buộc |
| **FR-3D-05** | **Mô phỏng tháo rời đầu đèn nam châm làm đèn pin cứu hộ** | Khi người dùng bật công tắc *"Tháo rời đầu đèn làm đèn pin"* trên bảng điều khiển (đặc biệt trong kịch bản mất điện và nước ngập), phần đầu đèn trên mô hình 3D sẽ tách rời khỏi đỉnh thân đèn và phát luồng sáng đèn pin. | Cao |
| **FR-3D-06** | **Hiển thị trạng thái Loa cảnh báo dạng Trực quan (Visual-Only)** | Không phát âm thanh thật qua trình duyệt. Thay vào đó, giao diện hiển thị **hiệu ứng sóng âm động (Visual Waveform)** kèm dòng thông điệp tương ứng:<br>- **Cấp 0**: Sóng âm tắt (*"Im lặng – Chế độ đèn trang trí"*).<br>- **Cấp 1**: Sóng âm nhẹ (*"Đang phát tiếng chuông gió báo hiệu mưa lớn & triều cường"*).<br>- **Cấp 2**: Sóng âm giọng nói (*"Đang phát giọng nói nhắc chủ nhà lắp ván chắn cửa, kê đồ"*).<br>- **Cấp 3**: Sóng âm đỏ cường độ cao (*"ĐANG HÚ CÒI BÁO ĐỘNG KHẨN CẤP"*), kèm nút **Tắt tiếng còi (Mute)** để chuyển trạng thái loa sang *Đã tắt còi (Muted)*. | Bắt buộc |

---

### 2.2. Phân hệ 2: Bảng Điều khiển Mô phỏng Môi trường & Cảm biến trên Trang Demo (`FR-SIM`)

| Mã YC | Tên chức năng | Mô tả chi tiết yêu cầu | Mức ưu tiên |
| :--- | :--- | :--- | :---: |
| **FR-SIM-01** | **Điều chỉnh nhóm thông số Thời tiết & Ngoại vi** | Cung cấp các thanh trượt (sliders) cho phép điều chỉnh:<br>- **Lượng mưa dự báo / hiện tại** ($\text{mm/h}$, từ $0$ đến $150\text{ mm/h}$).<br>- **Mức triều cường** ($\text{cm}$, từ $0$ đến $200\text{ cm}$).<br>- **Độ ẩm không khí** ($\%$, từ $40\%$ đến $100\%$).<br>- **Áp suất không khí** ($\text{hPa}$, từ $980$ đến $1020\text{ hPa}$). | Bắt buộc |
| **FR-SIM-02** | **Điều chỉnh nhóm thông số Cảm biến "Tàng hình" giả lập** | Cung cấp các thanh trượt thay thế cho tín hiệu từ các cảm biến vật lý:<br>- **Mực nước dưới cống/hố ga sân trước** ($\%$ độ đầy cống, từ $0\%$ đến $100\%$).<br>- **Tốc độ nước cống đang dâng** ($\text{cm/phút}$, từ $0$ đến $10\text{ cm/phút}$).<br>- **Mực nước chạm mặt sàn nhà** ($\text{cm}$, từ $0$ đến $50\text{ cm}$, phát hiện từ ngưỡng $1\text{ cm}$).<br>- **Thời gian nước đọng liên tục trên sàn** ($\text{giây}$, từ $0$ đến $60\text{ giây}$ – dùng để kiểm chứng bộ lọc chống báo động giả khi lau nhà). | Bắt buộc |
| **FR-SIM-03** | **Mô phỏng trạng thái Nguồn điện, Pin dự phòng & Mạng Wi-Fi** | Cung cấp các nút gạt (toggles) trên Trang Demo để giả lập:<br>- **Cắt điện lưới -> Chạy Pin sạc dự phòng (8–12 tiếng)**: Hiển thị dung lượng pin và thời gian duy trì ước tính.<br>- **Mất mạng Internet/Wi-Fi -> Chế độ hoạt động độc lập (Offline Standalone)**: Chứng minh đèn vẫn tự đo nước cống/sàn, chạy mô hình ML nội bộ và đổi màu báo động tại chỗ bình thường khi mất mạng. | Bắt buộc |
| **FR-SIM-04** | **Các nút Kịch bản mẫu 1-Click (Quick Scenario Presets)** | Cung cấp 5 nút chọn nhanh để tự động nạp bộ thông số chuẩn phục vụ thuyết trình chấm thi:<br>1. **Ngày bình thường (Cấp 0)**<br>2. **Dự báo sớm – Trước 60 phút (Cấp 1)**<br>3. **Nước cống dâng – Trước 15–30 phút (Cấp 2)**<br>4. **Ngập khẩn cấp tràn sàn lúc nửa đêm (Cấp 3)**<br>5. **Tình huống Lau nhà / Đổ nước (Test chống báo động giả)** | Bắt buộc |

---

### 2.3. Phân hệ 3: Mô hình Machine Learning Học Thích Nghi & Chống Báo động Giả (`FR-ML`)

| Mã YC | Tên chức năng | Mô tả chi tiết yêu cầu | Mức ưu tiên |
| :--- | :--- | :--- | :---: |
| **FR-ML-01** | **Huấn luyện nền từ dữ liệu mô phỏng lịch sử ngập TP.HCM** | Hệ thống khởi tạo và huấn luyện mô hình Machine Learning nền (Base Model) dựa trên bảng dữ liệu mô phỏng các trận ngập lịch sử tại TP.HCM (kết hợp giữa lượng mưa lớn, đỉnh triều cường, áp suất không khí và tốc độ nước cống dâng). | Bắt buộc |
| **FR-ML-02** | **Dự báo tức thì độ sâu ngập ($\text{cm}$) và xác suất ngập ($\%$)** | Ngay khi người dùng kéo thanh trượt trên Trang Demo, mô hình ML tính toán tức thì (`< 5ms`) xác suất ngập ($\%$) và **độ sâu ngập dự kiến trong 30–60 phút tới ($\text{cm}$)** dựa trên các thông số thời tiết, cảm biến cống và độ cao cốt nền nhà, đồng thời cập nhật ngay màu sắc và chiều cao cột sáng của đèn 3D. | Bắt buộc |
| **FR-ML-03** | **Học trực tuyến / Học tăng cường theo từng ngôi nhà (`partial_fit` & Reward Feedback)** | Mỗi ngôi nhà có độ cao cốt nền và đặc thù thoát nước khác nhau. Khi người dùng nhập **mức ngập thực tế sau cơn mưa ($\text{cm}$)** trên Trang Kết quả và nhấn **"Cập nhật thực tế (Dạy AI học)"**, mô hình sẽ tính sai số dự báo / điểm thưởng (Reward) và thực hiện bước học trực tuyến (`partial_fit` / Online Gradient Descent) để cập nhật trọng số ngay lập tức. | Bắt buộc |
| **FR-ML-04** | **Bộ lọc Chống Báo động Giả khi lau nhà hoặc đổ nước** | Hệ thống **chỉ kích hoạt trạng thái Khẩn cấp (Cấp 3)** khi thỏa mãn đồng thời 2 điều kiện:<br>1. Mực nước sàn $\ge$ ngưỡng tràn sàn (mặc định $1\text{ cm}$) **VÀ** Thời gian nước đọng liên tục $\ge$ ngưỡng thời gian lọc (mặc định $30\text{ giây}$).<br>2. Mực nước dưới cống đang đầy ($\ge$ ngưỡng Cấp 2) **HOẶC** Trời đang mưa ($\ge$ ngưỡng mưa).<br>Nếu có nước trên sàn nhưng thời gian đọng $< 30\text{ giây}$ hoặc dưới cống khô ráo & trời không mưa, hệ thống hiển thị nhãn *"Phát hiện nước sinh hoạt (Lau nhà / Đổ nước) – Không báo động giả"* và không bật còi/đèn đỏ. | Bắt buộc |

---

### 2.4. Phân hệ 4: Trang Kết quả Dự báo ML, Học Thích Nghi & Tinh chỉnh Ngưỡng (`FR-RES`)

| Mã YC | Tên chức năng | Mô tả chi tiết yêu cầu | Mức ưu tiên |
| :--- | :--- | :--- | :---: |
| **FR-RES-01** | **Hiển thị Báo cáo Dự báo Ngập & Trọng số Đóng góp từ Mô hình ML** | Trang Kết quả hiển thị trực quan đầu ra của mô hình ML dựa trên input từ Trang Demo:<br>- **Cấp độ cảnh báo dự báo** (Cấp 0 – Cấp 3), **Xác suất ngập ($\%$)** và **Độ sâu ngập dự kiến ($\text{cm}$)** trong 30–60 phút tới.<br>- **Biểu đồ đóng góp của từng yếu tố (Feature Contributions)**: Thể hiện rõ lượng mưa, triều cường, tốc độ nước cống dâng và độ cao cốt nền nhà đang tác động bao nhiêu $\text{cm}$ vào kết quả dự báo. | Bắt buộc |
| **FR-RES-02** | **Giao diện "Cập nhật Thực tế – Dạy AI Học" (Online Learning Panel)** | Cung cấp ô nhập **Mức ngập thực tế của nhà sau cơn mưa ($\text{cm}$)** và nút **"Cập nhật thực tế (Dạy AI học)"**. Sau khi bấm, giao diện hiển thị:<br>- Độ lệch trước khi học vs. sau khi học và điểm thưởng (Reward / độ chính xác cải thiện).<br>- Số lần (cơn mưa) mà AI đã học cho ngôi nhà này và dự báo mới sau khi cập nhật trọng số. | Bắt buộc |
| **FR-RES-03** | **Hiển thị Khu vực Ngập trên Mặt bằng & Checklist Khẩn cấp** | - Hiển thị trạng thái an toàn/ngập của từng khu vực trong nhà: *Sân trước / Hố ga*, *Phòng khách*, *Nhà vệ sinh / Bếp*.<br>- Hiển thị **Danh sách việc cần làm khẩn cấp (Emergency Action Checklist)** theo cấp độ ngập (lắp tấm chắn nước ở cửa, rút điện tủ lạnh, dắt xe lên cao, kê đồ nội thất...) cho phép tick hoàn thành, kèm nút **Tắt tiếng còi (Mute)**. | Bắt buộc |
| **FR-RES-04** | **Tinh chỉnh đầy đủ 3 nhóm Ngưỡng cấu hình (Threshold Settings)** | Cho phép người dùng/giảng viên chỉnh sửa các ngưỡng ngay trên Trang Kết quả và đồng bộ hai chiều với Trang Demo:<br>1. **Đặc điểm Ngôi nhà & Đèn**: Độ cao cốt nền nhà so với mặt đường/nắp cống ($\text{cm}$), Chiều cao thân đèn ($60\text{–}100\text{ cm}$).<br>2. **Ngưỡng Kích hoạt Cấp 1–2–3**: Ngưỡng lượng mưa ($\text{mm/h}$) & triều cường ($\text{cm}$) cho Cấp 1; Ngưỡng $\%$ nước cống đầy & tốc độ dâng ($\text{cm/phút}$) cho Cấp 2; Ngưỡng mực nước chạm sàn ($\text{cm}$, mặc định $1\text{ cm}$) cho Cấp 3.<br>3. **Ngưỡng Lọc Chống Báo giả**: Số giây nước đọng liên tục trên sàn tối thiểu để xác nhận ngập thật (mặc định $30\text{ giây}$). | Bắt buộc |

---

## 3. Yêu cầu Phi Chức năng (Non-Functional Requirements - NFR)

| Mã YC | Thuộc tính chất lượng | Mô tả yêu cầu chi tiết |
| :--- | :--- | :--- |
| **NFR-01** | **Hiệu năng Đồ họa 3D & Suy luận ML Tức thì** | - Mô hình [`resources/house.stl`](file:///home/thanhduy/Projects/du_an_den_canh_bao_ngap/resources/house.stl) (392 triangles) cùng chiếc đèn 3D và hiệu ứng mặt nước phải xoay mượt mà ở tốc độ $\ge 30\text{–}60\text{ FPS}$.<br>- Thời gian suy luận (inference) và cập nhật học trực tuyến (`partial_fit`) của mô hình ML phải đạt `< 10 ms`, đảm bảo phản hồi thời gian thực ngay khi kéo thanh trượt. |
| **NFR-02** | **Hoạt động Độc lập 100% (Zero External API Dependency)** | Mô hình Machine Learning chạy hoàn toàn nội bộ (Local/Edge), không phụ thuộc vào kết nối Internet hay khóa API trả phí bên ngoài, đảm bảo tính sẵn sàng tuyệt đối cho kịch bản cứu hộ khi mất mạng. |
| **NFR-03** | **Thẩm mỹ Giao diện Nội thất Sang trọng (UI/UX Aesthetics)** | Giao diện Web và mô hình đèn 3D phải thể hiện đúng tinh thần *"360 ngày là đèn trang trí sang trọng"*: tông màu tinh tế, hiệu ứng ánh sáng tán xạ dịu mắt (không lộ hạt LED thô), bố cục trực quan cho buổi bảo vệ đồ án. |

---

## 4. Ma trận Chuyển đổi Trạng thái Mô phỏng (Simulation State Matrix)

| Cấp độ | Điều kiện Đầu vào (Thanh trượt / Dự báo ML) | Đèn 3D (`smart_lamp_3d`) | Mặt nước 3D & Đèn Rọi Sàn | Trạng thái Loa (Visual-Only) | Trang Kết quả (ML & Checklist) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0. Ngày bình thường** | • Mưa & Triều cường $<$ Ngưỡng Cấp 1<br>• ML dự báo $0\text{ cm}$ ngập, sàn khô | **Vàng ấm** toàn thân (Đèn trang trí nội thất) | Mặt nước ẩn ($0\text{ cm}$)<br>Đèn rọi sàn: **Tắt** | Sóng âm tắt: *"Im lặng – Bình thường"* | Dự báo ML: **An toàn (0 cm)**<br>Không có việc khẩn cấp |
| **1. Dự báo sớm** *(Trước ~60p)* | • Lượng mưa hoặc Triều cường $\ge$ Ngưỡng Cấp 1 (hoặc ML dự báo nguy cơ ngập sớm) | **Xanh dương dịu**, hiệu ứng gợn sóng nhẹ | Mặt nước mấp mé cống<br>Đèn rọi sàn: **Tắt** | Sóng âm nhẹ: *"Chuông gió – Cảnh báo mưa lớn & triều cường"* | Dự báo nguy cơ ngập sớm 30–60p + Nhắc kiểm tra cửa & lịch triều cường |
| **2. Chuẩn bị ứng phó** *(Trước 15–30p)* | • Nước cống $\ge$ Ngưỡng Cấp 2 (hoặc tốc độ dâng cao)<br>• ML dự báo sắp ngập $H\text{ (cm)} > 0$ | **Màu Cam**, cột sáng dâng cao đúng bằng mức $H\text{ (cm)}$ dự báo | Mặt nước dâng ở sân trước<br>Đèn rọi sàn: **Tắt** | Sóng âm giọng nói: *"Nhắc lắp ván chắn cửa, kê cao đồ đạc"* | Hiển thị số $\text{cm}$ ngập dự kiến từ ML + Checklist chuẩn bị (lắp ván chắn, dắt xe) |
| **3. Khẩn cấp** *(Nước tràn sàn)* | • Nước chạm sàn $\ge$ Ngưỡng Cấp 3 (mặc định $1\text{ cm}$) **liên tục $\ge 30\text{s}$**<br>• **VÀ** (Cống đầy **HOẶC** Trời mưa) | **Chớp Đỏ** + Cột sáng hiển thị mức nước ngập | Mặt nước tràn vào nhà<br>Đèn trắng rọi sàn: **Bật sáng** | Sóng âm đỏ: *"ĐANG HÚ CÒI KHẨN CẤP"* (Có nút **Tắt còi / Mute**) | Báo đỏ khu vực ngập + Checklist khẩn cấp (rút điện tủ lạnh, sơ tán đồ) + Nút **Tắt còi** |
| **Ngoại lệ: Lau nhà / Đổ nước** | • Nước sàn $\ge 1\text{ cm}$ nhưng **thời gian đọng $< 30\text{s}$** **HOẶC** (Cống khô **VÀ** Không mưa) | Giữ nguyên trạng thái trước đó (Không báo đỏ) | Mặt nước ngập: Không kích hoạt<br>Đèn rọi sàn: **Tắt** | Không hú còi (*"Phát hiện nước sinh hoạt – Đã lọc báo giả"*) | Ghi nhận: *Nước sinh hoạt trên sàn (dưới ngưỡng lọc 30s hoặc cống khô ráo)* |
