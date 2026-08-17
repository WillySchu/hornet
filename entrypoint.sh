#!/bin/bash

as -o "$FILE".o "$FILE".s
gcc -o "$FILE" "$FILE".o
./"$FILE"; echo $?
