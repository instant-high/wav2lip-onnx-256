import torch
from wav2lip_256 import Wav2Lip

model = Wav2Lip()
checkpoint = torch.load('wav2lipv2.pth', map_location='cpu', weights_only=True)
s = checkpoint["state_dict"]
new_s = {}
for k, v in s.items():
    new_s[k.replace('module.', '')] = v
model.load_state_dict(new_s)
model = model.to('cpu')
model.eval()

x = torch.randn(1,1,80,16)
y = torch.randn(1,6,256,256)

torch.onnx.export(model, (x,y), 'wav2lip_256.onnx', input_names=['mel_spectrogram', 'video_frames'], output_names=['predicted_frames'], opset_version=15)

out = model(x,y)
print(out.shape)