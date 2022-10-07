FROM nvidia/cuda:11.4-devel-ubuntu20.04

ENV PROJECT=ff
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


ENV PYTHON_VERSION=3.8
ENV PYTORCH_VERSION=1.10.2+cu113
ENV TORCHVISION_VERSION=0.11.4+cu113
ENV CUDNN_VERSION=8.2.1.32-1+cuda11.4
ENV NCCL_VERSION=2.9.9-1+cuda11.4

ENV DEBIAN_FRONTEND=noninteractive


RUN mv /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda.list.bak
RUN echo "deb https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/ /" \
        >> /etc/apt/sources.list.d/cuda.list


# RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list

# RUN apt-get install --assume-yes apt-utils

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
        build-essential cmake g++ git curl docker.io vim wget ca-certificates libpcl-dev pybind11-dev

RUN apt-get install -y python${PYTHON_VERSION} python3-pip
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python & \
    ln -sf /usr/bin/pip3 /usr/bin/pip


RUN pip install --upgrade pip

RUN pip install \
        torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        -f https://download.pytorch.org/whl/torch_stable.html


RUN apt-get install -y --allow-change-held-packages \
        libcudnn8=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION}

RUN pip install tensorboard scikit-image

#RUN mkdir -p /home/xun.tong/ff
#WORKDIR /home/xun.tong/ff

#COPY . -r /home/xun.tong/ff

#RUN pip install -r /home/xun.tong/ff/requirements.txt


