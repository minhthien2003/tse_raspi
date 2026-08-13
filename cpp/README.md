# V49 Voice Lock — bản C++

Port đầy đủ của [`tse_run/voice_lock.py`](../tse_run/voice_lock.py) sang C++,
dùng cho Raspberry Pi 5 và về sau là NXP i.MX 95.

Phụ thuộc **duy nhất**: ONNX Runtime + ALSA. FFT, WAV I/O và NPY I/O đều tự
viết, không kéo theo thư viện thứ ba nào — quan trọng khi đóng image nhúng.

## Vì sao có bản C++

Không phải để nhanh hơn. ~99% thời gian mỗi chunk nằm trong ONNX Runtime,
vốn đã là C++. Lý do thật sự:

- Bản Python gọi ngược vào interpreter trong luồng audio realtime → GIL là
  rủi ro xrun. Bản C++ không có vấn đề đó.
- Image nhúng thường không muốn kèm Python runtime.
- Binary tĩnh, tích hợp thẳng vào sản phẩm.

Nếu chỉ cần giảm RTF trên Pi 5 thì `--power 1` hiệu quả hơn nhiều (~20%).

## Build

```bash
# 1. ONNX Runtime cho aarch64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/\
onnxruntime-linux-aarch64-1.20.1.tgz
tar xf onnxruntime-linux-aarch64-1.20.1.tgz

# 2. ALSA dev
sudo apt install -y libasound2-dev cmake build-essential

# 3. Build
cd tse_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release  -DONNXRUNTIME_ROOT=$PWD/../onnxruntime-linux-aarch64-1.20.1
cmake --build build -j4
```

Binary ra ở `build/tse_voice_lock`. CMake tự bật `-mcpu=cortex-a76` khi build
trên aarch64, và nhúng rpath nên không cần đặt `LD_LIBRARY_PATH`.

## Kiểm tra trước khi tin kết quả

```bash
./build/tse_voice_lock selftest
python tools/parity_ref.py --check ./build/tse_voice_lock
```

`selftest` không cần model, chỉ chạy STFT/iSTFT trên tín hiệu tất định.
`parity_ref.py` chạy cùng phép tính bằng numpy rồi so từng con số.

**Nếu FAIL thì đừng chạy model** — STFT sai thì output chỉ là rác, và nó sai
âm thầm chứ không báo lỗi.

Kết quả mong đợi (đã verify trên máy dev, g++ 6.3 x86):

| Giá trị | Python | C++ | Lệch |
|---|---|---|---|
| frames | 376 | 376 | 0 |
| roundtrip err | 1.79e-07 | 2.38e-07 | 6e-08 |
| sum(mag) | 75411.6 | 75411.6 | 1.6e-08 |
| mag[14,50] | 63.7361 | 63.7361 | 5.9e-08 |
| meanvar[1000] | 0.447213 | 0.447213 | 6.7e-08 |

Hai bản dùng thuật toán FFT khác nhau (numpy pocketfft vs radix-2 tự viết)
nên không thể bằng tuyệt đối; ngưỡng chấp nhận là 2e-5 tương đối.

## Dùng

Model tự dò trong `models/`, `../models/`, `../../models/` nên chạy từ đâu
cũng được:

```bash
cd build

./tse_voice_lock devices                       # liệt kê card ALSA
./tse_voice_lock enroll                        # tạo speaker_emb.npy
./tse_voice_lock bench                         # đo RTF
./tse_voice_lock live                          # realtime, ĐEO TAI NGHE
./tse_voice_lock file -i mixed.wav -o out.wav
./tse_voice_lock verify --emb-target me.npy --emb-other other.npy -i mixed.wav
```

`enroll` cần `models/wavlm_sv_int8.onnx` (tạo bằng
[`export_wavlm_onnx.py`](../tse_run/export_wavlm_onnx.py) trên laptop).
Không có nó thì enroll trên laptop rồi copy `speaker_emb.npy` sang.

### File config

Các giá trị hay dùng đặt trong [`voice_lock.conf`](../voice_lock.conf) ở gốc
project — dùng chung với bản Python, tự tìm ở `.`, `..`, `../..`:

```ini
model   = v49_int8.onnx
emb     = speaker_emb.npy
power   = 3
gain    = 3
lock_db = -8
threads = 4

in_device  = plughw:1,0
out_device = plughw:2,0
```

Thứ tự ưu tiên: **mặc định < `voice_lock.conf` < tham số dòng lệnh**. Nên vẫn
ghi đè được khi cần:

```bash
./tse_voice_lock live                # lấy hết từ config
./tse_voice_lock live --power 4      # như trên nhưng power = 4
./tse_voice_lock bench --no-config   # bỏ qua config
./tse_voice_lock live --config /etc/tse/pi5.conf
```

Gõ sai tên key sẽ **báo lỗi kèm số dòng**, không bỏ qua im lặng — key bị lơ
là cái bẫy khó phát hiện vì tưởng đã đổi cấu hình mà thực tế không.

Dòng `config:` khi khởi động cho biết file nào đang được dùng.

### Chọn card âm thanh

`./tse_voice_lock devices` in số card kèm khả năng thật ở 16 kHz mono:

```
  card 1: USB PnP Sound Device
           plughw:1,0  -> THU PHAT
```

Dùng `plughw:` chứ không dùng `hw:` — phần lớn mic USB không chạy 16 kHz
native, `plughw` để ALSA tự chuyển.

Nếu `default` hỏng (hay gặp trên Pi khi `asound.conf` khai báo `asym` mà
thiếu nhánh capture, lỗi `capture slave is not defined`), code tự dò card
khác và in ra nó chọn cái nào. Sửa tận gốc thì đặt `~/.asoundrc`:

```
pcm.!default {
    type asym
    playback.pcm "plughw:2,0"
    capture.pcm  "plughw:1,0"
}
```

## Cấu trúc

| File | Nội dung |
|---|---|
| `src/core.{h,cpp}` | FFT radix-2, STFT/iSTFT, chuẩn hóa. Không phụ thuộc gì |
| `src/io.{h,cpp}` | Đọc/ghi WAV và NPY tương thích numpy |
| `src/voice_lock.{h,cpp}` | Phiên ONNX, xử lý chunk, đo suppression |
| `src/enroll.{h,cpp}` | WavLM ONNX → embedding 512-d |
| `src/audio.{h,cpp}` | ALSA duplex |
| `src/main.cpp` | CLI |
| `tools/parity_ref.py` | Đối chiếu số học với bản Python |

## Điểm dễ sai khi sửa

**Cửa sổ Hann phải là periodic** `0.5 - 0.5·cos(2πn/N)`, không phải symmetric
(`np.hanning`). Training dùng `torch.hann_window` là periodic. Đây từng là bug
thật trong `backup/v49_pi5.py`.

**T cố định ở 376 frames.** Model export không có `dynamic_axes` nên chunk bắt
buộc đúng 3.0 s. Code đọc T từ graph thay vì hardcode, nhưng nếu re-export với
`dynamic_axes` thì phải kiểm tra lại đường xử lý shape động.

**Đo suppression phải bỏ `--gain` ra ngoài**, nếu không `--gain 3` sẽ làm lệch
ngưỡng lock đúng 3 dB.

**`NormalizeInput` cộng dồn ở double.** 96632 phần tử, cộng float32 sẽ trôi đủ
để lệch mean/std.

## Kiến trúc realtime

`live` chạy một vòng đồng bộ: đọc 1.5 s → xử lý → ghi 1.5 s. ALSA buffer đặt
4× period nên kernel vẫn thu tiếp trong lúc CPU chạy model. Điều kiện đúng đắn
là **thời gian xử lý < 1.5 s** — chạy `bench` để xác nhận trước.

Vượt ngưỡng đó sẽ sinh xrun; số xrun được in ra khi thoát.

## Chưa verify

Máy dev không có ONNX Runtime và ALSA, nên phần đã kiểm chứng thật sự là:

- `core.cpp` + `io.cpp`: **đã compile và chạy**, số khớp bản Python (bảng trên).
- `main.cpp`, `voice_lock.cpp`, `enroll.cpp`: **chỉ syntax-check** bằng header
  ONNX Runtime giả lập. Sạch, nhưng chưa link thật.
- `audio.cpp`: **chưa compile** — cần header ALSA.

Lần build đầu trên Pi có thể còn lỗi vặt. Chạy `selftest` và `parity_ref.py`
trước, rồi `bench`, rồi mới tới `live`.
