# GOT-OCR2.0-OpenVINO

<a href="https://huggingface.co/can-gaa-hou/GOT-OCR2.0-OpenVINO-INT4"><img src="https://img.shields.io/badge/Huggingface-yellow"></a>

Hi there! This repo shows how to use openvion to accelerate GOT-OCR2.0 model.

## Usage

1. Download all files from the [origin repo](https://huggingface.co/stepfun-ai/GOT-OCR2_0/tree/main) on huggingface, then move all files to the **weight** folder. The file structure will eventually look like this:
```
.
│  app.py
│  convert_model.py
├─ weight
│      config.json
│      generation_config.json
│      got_vision_b.py
│      modeling_GOT.py
│      qwen.tiktoken
│      render_tools.py
│      special_tokens_map.json
│      tokenization_qwen.json
│      tokenizer_config.json
```

2. Run the following command
```python
python app.py --image-file /path/to/image
```
It will automatically convert the model into OpenVINO IR using INT4 quantization. For more information about quantization with OpenVINO, please refer to [nncf](https://github.com/openvinotoolkit/nncf).


## Notes

1. Original version generates *19 Token/s*, while OV with INT4 quantiztion speed up to *37 Token/s* (Only test on Intel i7-1360P, 16GB, Windows 11 Pro).

2. Accuracy has not been tested yet, but it seems good to me.

3. Some code is generated from [ov_qwen2_audio_helper.py](https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/qwen2-audio/ov_qwen2_audio_helper.py).


## Acknowledgement

[GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0): Towards OCR-2.0 via a Unified End-to-end Model

[OpenVINO](https://github.com/openvinotoolkit/openvino): Open-source software toolkit for optimizing and deploying deep learning models.
