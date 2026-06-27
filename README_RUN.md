# DMS Web Package - Clean Video Overlay + Robust Head Nod

## Chạy web

```powershell
cd D:\drowsiness_detection-main\dms_web_train_package
python web_server.py --source webcam --decision-engine fsm
```

Mở trình duyệt:

```text
http://127.0.0.1:8000
```

## Cách dùng video

Trên web chọn video bằng nút **Choose File**, sau đó bấm **Upload & Start**.
Không cần nhập đường dẫn video thủ công.

## Thay đổi trong bản này

- Video không còn hiện đè các thông số EAR/MAR/PERCLOS/FPS.
- Video chỉ vẽ các phần nhận diện:
  - vùng mắt,
  - vùng miệng,
  - đường định hướng/góc đầu.
- Toàn bộ thông số realtime được đưa sang panel bên phải.
- Giao diện được nén để hạn chế phải cuộn màn hình; phần lý do cảnh báo hiện dạng tag nhỏ.
- Head nod đã được sửa để giảm cảnh báo sai do xe xóc:
  - không dùng pitch lớn đơn lẻ để kết luận gật đầu,
  - không dùng pitch velocity đơn lẻ để leo thang trạng thái,
  - chỉ kích hoạt head nod khi có đủ 3 điều kiện: góc đầu phù hợp, có chuyển động đầu gần đây, và có ngữ cảnh buồn ngủ từ mắt/PERCLOS.

## Dừng chương trình

Trong PowerShell nhấn:

```powershell
Ctrl + C
```


## Chạy cho điện thoại / máy khác trong cùng Wi-Fi

Bản này mặc định chạy Flask với `host=0.0.0.0`, nên các thiết bị cùng mạng Wi-Fi có thể truy cập bằng IP LAN của máy tính.

1. Chạy server:

```powershell
python web_server.py --source webcam --decision-engine fsm
```

2. Xem dòng terminal dạng:

```text
[WEB] Network: http://192.168.x.x:8000
```

3. Trên điện thoại cùng Wi-Fi, mở địa chỉ đó trong Chrome/Safari.

Nếu không vào được, mở Windows Defender Firewall và cho phép Python truy cập Private network, hoặc chạy:

```powershell
netsh advfirewall firewall add rule name="DMS Web 8000" dir=in action=allow protocol=TCP localport=8000
```

## Link public qua Internet

Cách đơn giản khi demo là dùng ngrok:

```powershell
ngrok http 8000
```

Ngrok sẽ tạo một URL HTTPS public trỏ về web local. Chỉ dùng để demo, không nên upload dữ liệu nhạy cảm khi mở public link.

## Âm thanh cảnh báo

File `static/alert.wav` được phát trong trình duyệt khi trạng thái là `DROWSY` hoặc `CRITICAL`. Trên điện thoại, trình duyệt thường yêu cầu người dùng bấm vào trang một lần trước khi cho phép phát âm thanh.
