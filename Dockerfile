FROM nvcr.io/nvidia/pytorch:23.09-py3

# Sber ML-Space params !!! DO NOT TOUCH IT !!!
USER root
# Add "jovyan"
RUN groupadd -g 1000 jovyan
RUN useradd -g jovyan -u 1000 -m jovyan
RUN mkdir -p /tmp/.jupyter_data && chown -R jovyan /tmp/.jupyter_data && \
    mkdir -p /tmp/.jupyter && chown -R jovyan /tmp/.jupyter
RUN mkdir -p /home/user && chown -R jovyan /home/user



### Install Linux dependencies! YOU CAN TWEAK IT!
USER root
RUN apt-get update --fix-missing && apt-get upgrade -y  &&\
    echo "8" apt-get install -y software-properties-common && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install tzdata -qy &&\
    apt install -qy \
        python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        zip \
        unzip \
        unrar \
        yasm \
        python3-dev \
        nano \
        vim \
    	git-lfs \
	htop \
        neovim \
        pkg-config \
        ffmpeg \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio &&\
    apt upgrade -qy &&\
    apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

### Install my Python dependencies! YOU CAN TWEAK IT!
RUN apt update && pip3 install  -U --no-cache-dir \
    openmim \
    opencv-python-headless \
    timm \
    mmdet \
    termcolor \
    yacs \
    pyyaml \
    scipy \
    pyyaml \
    huggingface_hub \
    safetensors \
    Pillow \
    psutil \
    PyYAML \
    requests \
    thop \
    tqdm \
    matplotlib
# Pytorch

RUN pip3 install -U uvicorn==0.23.2 peft==0.5.0 bitsandbytes==0.41.1 transformers==4.34.1
RUN pip3 install -U fastapi==0.103.2

COPY . /app
WORKDIR /app

ARG HF_TOKEN
ARG MODEL_NAME

RUN huggingface-cli download --token=${HF_TOKEN} ${MODEL_NAME}
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
