import torch
from transformers import AutoModelForSpeechSeq2Seq
device = "cuda:0"
model_id = "openai/whisper-large-v3"
modelSR = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=True
    )
modelSR.to(device)