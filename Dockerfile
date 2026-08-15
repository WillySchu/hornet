FROM --platform=linux/amd64 ubuntu:latest

RUN apt-get update
RUN apt-get install -y nasm build-essential gdb
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY test.s test.s
COPY entrypoint.sh entrypoint.sh
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ./entrypoint.sh
