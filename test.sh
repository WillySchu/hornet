#!/bin/bash

Docker build -t test .
docker run test
