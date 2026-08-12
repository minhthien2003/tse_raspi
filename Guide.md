
# TRÊN LAPTOP — chạy 1 lần
pip install torch transformers onnx onnxruntime
python export_wavlm_onnx.py --outdir models
# -> models/wavlm_sv_int8.onnx (~95 MB)

# copy sang Pi 5
scp "models/v49_int8 1.onnx" models/wavlm_sv_int8.onnx pi@raspi:~/tse_run/models/

# TRÊN PI 5 — chỉ 4 package, không torch
pip install onnxruntime numpy sounddevice soundfile

python voice_lock.py enroll --seconds 10 --encoder models/wavlm_sv_int8.onnx
python voice_lock.py live --model "models/v49_int8 1.onnx" --emb speaker_emb.npy --power 3
--encoder có thể bỏ — script tự tìm models/wavlm_sv_int8.onnx rồi wavlm_sv_fp32.onnx. Nếu không thấy cả hai thì mới fallback sang torch, và nếu cũng không có torch thì báo lỗi kèm đúng 2 cách xử lý.
