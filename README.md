# wav2lip-onnx-256x256 model
Simple and fast wav2lip using new 256x256 resolution trained onnx-converted model for inference

Minimum version. No additional functions like face enhancement, face alignment. Just same functions as the original repository

Inference is quite fast running on CPU using the converted wav2lip onnx models and antelope face detection. Can be run on Nvidia GPU, tested on RTX3060 Update: tested on GTX1050

Some result:

wav2lip 96x96  -  wav2lip_gan 96x96  -  wav2lip 256x256

https://github.com/user-attachments/assets/bdd186f6-6a79-4cbd-824f-74108392d390

* Installation: Clone this repository and read Setup.txt

* Download models from releases.

* Don't forget to install ffmpeg and set path variable.

* Face detection checkpoint already in insightface_func/models/antelope

Original 256x256 pretrained checkpoint taken from:

https://github.com/Kedreamix/Linly-Talker/blob/main/README.md


