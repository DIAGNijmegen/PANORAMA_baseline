
#FROM --platform=linux/amd64 pytorch/pytorch
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base
 
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)
 
# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1
ENV PYTHONWARNINGS="ignore"
 
# Install required system dependencies
RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  wget \
  curl \
  unzip \
  libopenblas-dev \
  python3.11 \
  python3.11-dev \
  python3.11-venv \
  python3.11-distutils \
  nano \
  build-essential \
  gfortran \
  gcc \
  g++ \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  python3-numpy \
  python3-scipy \
  python3-pillow \
  libx11-dev \
  libxext-dev \
  libxrender-dev \
  libsm6 \
  && apt-get clean autoclean && \
  apt-get autoremove -y && \
  rm -rf /var/lib/apt/lists/*
 
# Ensure python3 defaults to Python 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11
 
# Install pip manually
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
 
# Verify pip installation
RUN python3 -m pip --version
 
# Use pip to install Cython
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install cython
 
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
 
RUN mkdir -p /opt/algorithm
RUN chown -R user /opt/algorithm
ENV PATH="/home/user/.local/bin:${PATH}"
 
USER user
 
COPY --chown=user:user requirements.txt /opt/app/
# Ensure required Python packages for building scikit-image are installed
RUN python3 -m pip install --upgrade pip setuptools wheel cython numpy
 
# You can add any Python dependencies to requirements.txt
RUN python3 -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt
 
# RUN pip3 install --no-cache-dir "numpy<1.27.0"
# RUN pip3 install --no-cache-dir blosc2 graphviz onnx
 
### Clone nnUNet
# Configure Git, clone the repository without checking out, then checkout the specific commit
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/
 
# Install a few dependencies that are not automatically installed
RUN pip3 install \
        -e /opt/algorithm/nnunet \
        graphviz \
        onnx \
        SimpleITK && \
    rm -rf ~/.cache/pip
 
COPY --chown=user:user ./src/customTrainerCEcheckpoints.py /opt/algorithm/nnunet/nnunetv2/training/nnUNetTrainer/customTrainerCEcheckpoints.py
COPY --chown=user:user ./src/nnUNet_results/ /opt/algorithm/nnunet/nnUNet_results/
 
### Define workdir
WORKDIR /opt/app
 
COPY --chown=user:user ./src/process.py /opt/app/
COPY --chown=user:user ./src/data_utils.py /opt/app/
COPY --chown=user:user ./src/__init__.py /opt/app/
 
### Set environment variable defaults
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"
 
ENTRYPOINT [ "python3", "-m", "process" ]
