#!/bin/bash

# Fail if there is an error
set -e
set -o pipefail


# Adds a directory to path if it exits
pathadd(){
  if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
    PATH="${PATH:+"$PATH:"}$1"
  fi
}

pythonpathadd() {
  if [ -d "$1" ] && [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
    PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$1"
  fi
}

# Needs to add path to FastLAS for Condor
pathadd "/homes/rp218/bin"
export PYTHONPATH=${PYTHONPATH}:"/vol/bitbucket/rp218/luke-for-roko"

source /vol/bitbucket/rp218/thesis-venv/bin/activate;

# IMPORTANT: this number must match the queue number in condor.cmd
process_id=${1}

rev_digits=()
for (( i = ${#1} - 1; i >= 0; --i )); do
  rev_digits+=(${1:$i:1})
done

while [ ${#rev_digits[@]} -lt 3 ];
do
  rev_digits+=(0)
done

repeats=$((${rev_digits[0]}+1))
noise_pct=$((${rev_digits[1]}*10))

[[ ${rev_digits[2]} -lt 2 ]] && num_digits=4 || num_digits=9

args=(
 --num-digits ${num_digits}
 --noise-pct ${noise_pct}
 --repeat-num ${repeats}
)
if [ $(expr ${rev_digits[2]} % 2) != "0" ]; then
  args+=(--generator-nn)
fi

python3 sudoku_train.py ${args[@]}
