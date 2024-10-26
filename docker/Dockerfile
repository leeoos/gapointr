# Set the base CUDA image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 

# Install Python, Git, and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # llvm \
    # llvm-dev \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Set LLVM_CONFIG environment variable
ENV LLVM_CONFIG=/usr/bin/llvm-config


# Set Python3 as the default python command
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# RUN pip3 install --index-url http://host.docker.internal:8080/simple torch torchvision torchaudio

# Set CUDA architecture list to match your target GPU 
# ENV TORCH_CUDA_ARCH_LIST="7.5"

# Set the working directory to /PoinTr
WORKDIR /PoinTr

# Copy your local repository to the container
COPY . /PoinTr

# Install dependencies from requirements.txt
RUN pip install -r /PoinTr/requirements.txt --timeout 10000

# Install PoinTr dependencies
RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Manually install other dependencies in 'extensions'
# RUN . install.sh

# # Install clifford module
# RUN pip install git+https://github.com/pygae/clifford.git


# Start on PoinTr root dir
WORKDIR /PoinTr/