FROM pytorch/pytorch:latest
ENV NCCL_SOCKET_IFNAME IBNet
ENV NCCL_DEBUG INFO
NCCL_DEBUG_SUBSYS ALL
MAINTAINER Oussama Saoudi
WORKDIR /data
COPY . /data
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

