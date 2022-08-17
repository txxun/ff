FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV PROJECT=ff
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


ENV PYTHON_VERSION=3.8
ENV PYTORCH_VERSION=1.10.0+cu113
ENV TORCHVISION_VERSION=0.11.1+cu113
ENV CUDNN_VERSION=8.2.1.32-1+cuda11.3
ENV NCCL_VERSION=2.9.9-1+cuda11.3

RUN mv /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda.list.bak
RUN echo "deb https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/ /" \
        >> /etc/apt/sources.list.d/cuda.list


RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
        build-essential cmake g++-4.8 git curl docker.io vim wget ca-certificates

RUN apt-get install -y python${PYTHON_VERSION} python3-pip
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip


RUN pip install --upgrade pip

RUN pip install \
        torch==${PYTORCH_VERSION} \
        torchvision==${PYTORCHVISION_VERSION} \
        -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install -y \
        libcudnn8=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION}

RUN pip install tensorboard pybind11

RUN apt-get install pcl


