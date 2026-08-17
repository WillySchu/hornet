#!/bin/bash

python3 codegen.py --platform linux "$1".ht > "$1".s
Docker build -t test --build-arg FILE="$1" .
docker run test
