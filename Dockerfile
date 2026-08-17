FROM --platform=linux/amd64 ubuntu:latest

RUN apt-get update
RUN apt-get install -y nasm build-essential gdb
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /

ARG FILE=test.s

COPY ${FILE}.s ${FILE}.s
COPY entrypoint.sh entrypoint.sh
RUN chmod +x ./entrypoint.sh

ENV FILE=${FILE}

ENTRYPOINT ["./entrypoint.sh"]
