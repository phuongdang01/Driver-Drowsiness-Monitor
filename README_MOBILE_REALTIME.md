# DMS Mobile Realtime

Bản này bổ sung chế độ `Mobile realtime` để người khác mở link HTTPS trên điện thoại/laptop, bật camera của chính thiết bị đó và chạy phát hiện buồn ngủ theo phiên riêng.

## Chạy local hoặc qua Cloudflare Tunnel

```powershell
cd D:\drowsiness_detection-main\dms_web_train_package
python web_server.py --source webcam --decision-engine fsm
```

Mở trên máy chạy server:

```text
http://127.0.0.1:8000/mobile
```

Muốn gửi link public tạm thời:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Sau đó mở link Cloudflare và vào:

```text
https://xxxxx.trycloudflare.com/mobile
```

## Lưu ý quan trọng

- Trình duyệt chỉ cho phép dùng camera với `HTTPS` hoặc `localhost`.
- Mỗi người dùng có `session_id` riêng trong trình duyệt, nên EAR threshold, MAR threshold, FSM state, PERCLOS, blink/yawn không dùng chung.
- Chế độ này gửi frame từ trình duyệt lên server rồi server xử lý bằng MediaPipe + FSM. Nhiều người dùng cùng lúc sẽ cần máy chủ khỏe hơn.
- Nếu deploy cloud thật, dùng server production như Gunicorn thay vì Flask development server.

## Đường dẫn

- `/` : giao diện cũ cho webcam/video upload trên máy chủ.
- `/mobile` : giao diện realtime camera từ trình duyệt điện thoại/laptop.
