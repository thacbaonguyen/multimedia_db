# 📋 Hướng dẫn Thu thập Dữ liệu Âm thanh (Dành cho Team Member)

**Dự án:** Hệ CSDL lưu trữ và tìm kiếm tiếng động vật (INT1418)  
**Giai đoạn 1:** Xây dựng Dataset (Tối thiểu 500 files)

Chào bạn, đây là tài liệu hướng dẫn chi tiết các tiêu chí và cách thức thu thập dữ liệu âm thanh để bạn thực hiện. Dữ liệu này là "nguyên liệu" đầu vào quan trọng nhất, quyết định độ chính xác của hệ thống AI sau này.

---

## 🎯 1. Mục tiêu công việc
Sưu tầm và phân loại **ít nhất 500 files âm thanh** ngắn về tiếng kêu của các loài động vật khác nhau, sau đó sắp xếp chúng vào đúng cấu trúc thư mục của dự án.

## ⚠️ 2. Tiêu chuẩn khắt khe đối với File âm thanh
Để AI Agent có thể xử lý và trích xuất đặc trưng chính xác, các file thu thập về **BẮT BUỘC** phải thỏa mãn các điều kiện sau:

1. **Sự thuần khiết (Chỉ 1 loài):** 
   - Mỗi file chỉ chứa tiếng kêu của **DUY NHẤT 1 loài động vật**. 
   - Không được có người nói chuyện xen vào, không có nhạc nền, và không có tiếng động cơ/xe cộ quá ồn át tiếng động vật.
2. **Độ dài phù hợp:** 
   - Không dùng file quá ngắn (< 1 giây) vì không đủ dữ liệu trích xuất đặc trưng.
   - Không dùng file quá dài (> 20 giây sẽ làm phình dung lượng và xử lý chậm). 
   - **Tốt nhất là từ 2 giây đến 10 giây/file**.
3. **Đa dạng loài (Classes):** 
   - Nên thu thập từ 5 đến 10 loài khác nhau (ví dụ: Chó, Mèo, Gà, Ếch, Chim, Bò, Heo...). 
   - Phân bổ số lượng đồng đều (ví dụ: thu thập 10 loài, mỗi loài 50 files = 500 files).
4. **Định dạng file:**
   - Khuyên tải về định dạng `.wav` hoặc `.mp3`. AI định dạng hệ thống sau đó sẽ tự động chuẩn hoá lại, nhưng đầu vào càng chất lượng thì kết quả nhận diện càng cao.

---

## 📁 3. Cách thức bàn giao Dữ liệu (Cấu trúc thư mục)
Sau khi tải file về, bạn cần phân loại chúng ngay lập tức. Hãy tạo thư mục theo cấu trúc dưới đây bên trong máy của bạn (hoặc trên Google Drive dùng chung của nhóm):

```text
data_giao_nop/
├── cat (thư mục chứa toàn bộ tiếng mèo)
│   ├── meo_01.wav
│   ├── meo_02.mp3
│   └── ...
├── dog (thư mục chứa toàn bộ tiếng chó)
│   ├── cho_01.wav
│   └── ...
├── rooster (tiếng gà gáy)
└── ... (các loài khác)
```
*Lưu ý: Tên thư mục loài nên viết bằng tiếng Anh không dấu viết thường (cat, dog, bird, frog...) để hệ thống AI code sau này dễ đọc nhãn (label).*

---

## 🔍 4. Gợi ý Nguồn tài nguyên dễ lấy Data nhất
Thay vì phải tự ghi âm hoặc download từng cái trên YouTube, bạn hãy lên **Kaggle** để lấy các Dataset đã được cộng đồng dọn dẹp sẵn (chỉ cần đăng ký 1 tài khoản Kaggle miễn phí khoản 1 phút).

*   **ESC-50 Dataset:** Đây là bộ dữ liệu âm thanh môi trường cực kỳ nổi tiếng. Trong đó có chứa đúng 500 files âm thanh của 10 loài động vật (Dog, Rooster, Pig, Cow, Frog, Cat, Insects, Sheep, Crow, Rain/Birds). Tải [ESC-50 trên Kaggle](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50).
*   **Animal Sounds Dataset:** Các bộ data chuyên về động vật trên Kaggle. Bộ này có [Animal sounds (Kaggle)](https://www.kaggle.com/datasets/vencerlanz09/animal-sounds) phù hợp sẵn cho việc phân loại.
*   **Freesound.org:** Nếu còn thiếu file, bạn có thể gạch từ khóa trên freesound.org (như "dog bark", "cat meow") để tải thêm file bổ sung.

---

## 🚀 5. Checklist hoàn thành nhiệm vụ
Trước khi bàn giao lại cho người chạy AI Agent, hãy kiểm tra:
- [ ] Tổng số file đếm thử đã >= 500 chưa?
- [ ] Các thư mục đã đại diện cho loài vật chưa (cat, dog...)?
- [ ] Nghe thử ngẫu nhiên 5 file xem có bị dính tiếng người nói/nhạc nền hay không?
- [ ] Nén toàn bộ folder `data_giao_nop` thành file `.zip` và gửi lại cho trưởng nhóm.
