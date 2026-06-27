# Stage 1 - Start / Stop / Continue

Bản này bổ sung luồng điều khiển phiên realtime:

- **Start camera**: bắt đầu một phiên mới, reset thuật toán và hiệu chuẩn lại EAR/MAR.
- **Stop**: tạm dừng phiên hiện tại, không xoá ngưỡng đã hiệu chuẩn, không xoá lịch sử sự kiện.
- **Continue**: chạy tiếp phiên vừa tạm dừng, giữ nguyên ngưỡng EAR/MAR, FSM, PERCLOS, blink/yawn và lịch sử.
- **Reset calibration**: tính lại ngưỡng trong phiên đang chạy.

Với camera, khi bấm Stop, trình duyệt dừng camera để giảm tiêu thụ tài nguyên và quyền riêng tư. Khi bấm Continue, hệ thống xin/mở lại camera và tiếp tục xử lý với trạng thái hiện tại.

Với video file, Stop sẽ pause video tại vị trí hiện tại, Continue chạy tiếp từ đúng vị trí đó.
