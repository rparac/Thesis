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

# IMPORTANT: this number must match the queue number in condor.cmd
n_of_folds=10
fold_id=${1}
is_in_labs=${2}
# Change depending on the whether we are doing atomisation/generalisation
#problem_type="atomisation"
problem_type="generalisation"
base_dir="ilasp/${problem_type}"
background_file_loc="${base_dir}/background.ilasp"
lang_bias_loc="${base_dir}/generated_search_space.txt"
example_file_template="${base_dir}/ilasp_examples/examples_%s.ilasp"
output_file="${base_dir}/solutions/sol_${fold_id}.ilasp"
pylasp_script="${base_dir}/measurements_pylasp_script.py"
#output_file="${base_dir}/solutions/hand_made_sol.lp"

training_files=()
for ((i=0; i<${n_of_folds};i++)); do
  if [[ ${i} -ne ${fold_id} ]]; then
    training_files+=($(printf ${example_file_template} ${i}))
  fi
done
eval_file=$(printf ${example_file_template} ${fold_id})


# Needs to add path to ILASP for Condor
pathadd "/homes/rp218/bin"

# Delete possible an old output file
[ -e ${output_file} ] && rm ${output_file}

# Run memory measuring job
nohup ./memory_script.sh >> ${output_file} &

#flags=("--version=4" "--small-constraints" "--restarts")
#flags=("--quiet" "--version=3")
flags=("--debug" "--restarts")
ILASP ${flags[@]} ${background_file_loc} ${lang_bias_loc} ${training_files[@]} ${pylasp_script} >> ${output_file}

# Run evaluation
if ! [ -z ${is_in_labs} ]; then
  source /vol/bitbucket/rp218/thesis-venv/bin/activate;
else
  source venv/bin/activate;
fi
# Download the spacy language pack in use
python3 -m spacy download en_core_web_lg > /dev/null

# PYTHONPATH must know where is the current directory
export PYTHONPATH=${PYTHONPATH}:$(pwd)
args=(
  --evalexample ${eval_file}
  --sol ${output_file}
)

if ! [ -z ${is_in_labs} ]; then
  args+=(--labs)
fi
if [[ ${problem_type} == "atomisation" ]]; then
  args+=(--atomisation)
fi

python3 cmd/evaluate.py ${args[@]}
