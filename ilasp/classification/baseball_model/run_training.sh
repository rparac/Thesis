#!/bin/bash

# Adds a directory to path if it exits
pathadd(){
  if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
    PATH="${PATH:+"$PATH:"}$1"
  fi
}

example_id=${1}

# Needs to add path to ILASP for Condor
pathadd "/homes/rp218/bin"

FastLAS --nopl background.lp language_bias.ilasp examples_${example_id}.las --debug
