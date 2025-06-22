FROM python:3.10.14 as python_img
FROM ubuntu:22.04
COPY --from=python_img /usr/local/bin/python /usr/local/bin/python
COPY --from=python_img /usr/local/lib /usr/local/lib
COPY --from=python_img /usr/local/include /usr/local/include

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y build-essential --no-install-recommends make \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    tmux \
    cloc \
    cmake \
    libopenmpi-dev \
    locales \
    software-properties-common \
    gpg-agent \
    ffmpeg \
    libsm6 \
    libxext6 \
    jq \
    git

# just
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /root/.cargo/bin

# pkl
WORKDIR /bin
RUN curl -L -o pkl https://github.com/apple/pkl/releases/download/0.25.2/pkl-linux-amd64 && chmod +x pkl

# dependencies
WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
RUN python -m pip install guardrails-ai==0.5.13
# replace the your guardrail token here, if close-sourced models are not needed, comment out the following two lines
RUN guardrails configure --disable-metrics --disable-remote-inferencing --token xxxxx
RUN guardrails hub install hub://guardrails/valid_choices

# # set up entry point
ENTRYPOINT ["/root/.cargo/bin/just"]
