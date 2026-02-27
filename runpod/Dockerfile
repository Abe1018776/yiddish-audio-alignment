FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

RUN apt update && apt install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    runpod \
    stable-ts>=2.16.0 \
    faster-whisper>=1.0.0 \
    torch==2.4.1 \
    huggingface-hub>=0.36.0 \
    numpy \
    tqdm

# Bake the Yiddish whisper model into the image — no runtime download
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/yi-whisper-large-v3-turbo-ct2"); print("Yiddish model cached!")'

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
