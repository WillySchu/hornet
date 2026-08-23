#!/bin/bash

set -e

python3 compile.py --platform linux "$1".ht > "$1".s
Docker build -t test --build-arg FILE="$1" .
docker run test
