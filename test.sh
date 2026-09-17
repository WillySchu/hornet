#!/bin/bash

set -e

python3 build.py "$1".ht --output "$1"
./"$1"
rm "$1"
