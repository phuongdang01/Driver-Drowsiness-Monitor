# DMS Browser MediaPipe + Server Upload

Bản này giữ cả hai chế độ:

1. `/` hoặc `/mobile`: camera/video local chạy MediaPipe trực tiếp trên trình duyệt. Video không upload lên server. Phù hợp triển khai realtime public.
2. `/upload`: chế độ upload video lên server như các bản cũ. Server dùng OpenCV/MediaPipe Python để đọc video, hỗ trợ tốt hơn với AVI/dataset.

Lưu ý: Video AVI thường không chạy ổn trong thẻ video HTML5 của trình duyệt. Muốn test AVI, hãy dùng `/upload`. Muốn xử lý local trên trình duyệt, nên dùng MP4/H.264 hoặc MOV.

Chạy local:

```powershell
python web_server.py
```

Mở realtime browser:

```text
http://127.0.0.1:8000/
```

Mở upload server-side:

```text
http://127.0.0.1:8000/upload
```
