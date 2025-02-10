import argparse
import torch
import requests
import dataclasses
import nncf
from PIL import Image
from io import BytesIO
from typing import List
from enum import auto, Enum
from convert_model import OVGotOcrModel
from transformers import AutoTokenizer, TextStreamer, StoppingCriteria
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from huggingface_hub import snapshot_download


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "<|im_end|>"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            if self.system:
                ret = self.system + self.sep 
            else:
                ret = ''
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")


    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False
    

class GOTImageEvalProcessor:
    def __init__(self, image_size=384, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
    def __call__(self, item):
        return self.transform(item)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(image_file, model, tokenizer):

    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
    DEFAULT_IM_START_TOKEN = '<img>'
    DEFAULT_IM_END_TOKEN = '</img>'
    # Model

    # TODO vary old codes, NEED del
    image_processor = GOTImageEvalProcessor(image_size=1024)

    use_im_start_end = True

    image_token_len = 256

    image = load_image(image_file)

    qs = 'OCR: '

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs



    conv_mpt = Conversation(
        system="""<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail.""",
        # system = None,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        version="mpt",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
    )

    conv = conv_mpt.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    inputs = tokenizer([prompt])

    image_tensor = image_processor(image)

    input_ids = torch.as_tensor(inputs.input_ids).cpu()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    import time

    start = time.time()
    output_ids = model.generate(
        input_ids,
        images= [image_tensor.unsqueeze(0).cpu()],
        do_sample=False,
        num_beams = 1,
        no_repeat_ngram_size = 20,
        streamer=streamer,
        max_new_tokens=4096,
        stopping_criteria=[stopping_criteria],
        )
    end = time.time()
    print(f"\n Generate time {end - start}s")

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", type=str, default="./weight")
    parser.add_argument("--image-file", type=str, required=True)
    args = parser.parse_args()
    model_dir = args.weight_dir
    snapshot_download(
        repo_type="model",
        repo_id="stepfun-ai/GOT-OCR2_0",
        local_dir=model_dir,
        revision="main",
        local_dir_use_symlinks=False,
    )
    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 128,
        "ratio": 1.0,
    }
    model = OVGotOcrModel(model_dir, "CPU", compression_configuration=compression_configuration)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    with torch.no_grad():
        eval_model(args.image_file, model, tokenizer)