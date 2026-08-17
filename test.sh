#!/bin/bash

Docker build -t test --build-arg FILE="$1" .
docker run test
