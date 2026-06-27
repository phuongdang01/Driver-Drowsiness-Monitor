# Deploy public độc lập trên Render

Mục tiêu: tạo link HTTPS public để người khác mở `/mobile`, dùng camera của chính điện thoại/laptop của họ và xử lý theo session riêng.

## 1. Chuẩn bị GitHub

1. Giải nén file zip.
2. Mở thư mục `dms_web_train_package`.
3. Tạo repository mới trên GitHub.
4. Upload toàn bộ file BÊN TRONG thư mục `dms_web_train_package` lên repo root.

Repo root cần có các file chính:

- `web_server.py`
- `requirements.txt`
- `Procfile`
- `render.yaml`
- thư mục `runtime/`
- thư mục `templates/`
- thư mục `static/`

## 2. Deploy trên Render

Cách 1: dùng Blueprint nếu Render nhận `render.yaml`.

Cách 2: tạo Web Service thủ công:

- Environment: Python
- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
gunicorn web_server:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

## 3. Link sử dụng

Sau khi deploy xong, Render sẽ cấp link dạng:

```text
https://ten-app.onrender.com
```

Trang mobile realtime:

```text
https://ten-app.onrender.com/mobile
```

Người dùng mở link này, bấm `Start camera`, cấp quyền camera và hệ thống sẽ chạy theo session riêng của người đó.

## 4. Ghi chú

- Camera trình duyệt cần HTTPS, Render cung cấp HTTPS public.
- Nhiều người dùng cùng lúc sẽ cần CPU/RAM mạnh hơn.
- Bản miễn phí có thể khởi động chậm sau thời gian không dùng.
