# a library of helper functions for scripting
# shell scripts should source this library.

# the highest level commands in here are:  run_gpus round_robbin_gpu run fork


# Helper function to ensure only one instance of a job runs at a time.
# Optionally, on finish, can write a file to ensure the job won't run again.
# usage: use_lockfile myfile.locked [ myfile.finished ]
function use_lockfile() {
  lockfile_fp="$(realpath -m ${1}.running)"
  lockfile_runonce="$2"
  lockfile_success_fp="$(realpath -m ${1}.finished)"
  lockfile_failed_fp="$(realpath -m ${1}.failed)"
  # create lock file
  if [ -e "$lockfile_fp" ] ; then
    echo "job already running!"
    exit
  fi
  if [ "$lockfile_runonce" = "yes" -a -e "$lockfile_success_fp" ] ; then

    local YELLOW='\033[0;33m'
    local NC='\033[0m' # No Color
    echo -e "$YELLOW job previously completed!$NC  $lockfile_success_fp"
    exit
  fi
  mkdir -p "$(dirname "$lockfile_fp")"
  runid=$RANDOM
  echo $runid > "$lockfile_fp"

  # check that there wasn't a race condition
  # (not guaranteed to work but should be pretty good)
  sleep $(bc -l <<< "scale=4 ; ${RANDOM}/32767/10")
  rc=0
  grep $runid "$lockfile_fp" || rc=1
  if [ "$rc" = "1" ] ; then
    echo caught race condition 
    exit 1
  fi

  # before starting current job, remove evidence of failed job.
  if [ -e "$lockfile_failed_fp" ] ; then
    rm "$lockfile_failed_fp"
  fi

  # automatically remove the lockfile when finished, whether fail or success
  function remove_lockfile() {
    rm $lockfile_fp
  }
  function trap_success() {
    local GREEN='\033[0;32m'
    local NC='\033[0m' # No Color
    if [ ! -e "$lockfile_failed_fp" ] ; then
      if [ "$lockfile_runonce" = "yes" ] ; then
        echo -e "$GREEN $run_id :  success! \n$NC    ...to re-run job, rm this file: ${lockfile_success_fp}"
        date > $lockfile_success_fp
        hostname >> $lockfile_success_fp
      else
        echo success $run_id
      fi
    fi
    remove_lockfile
    exit 0
  }
  function trap_err() {
    rv=$?
    local RED='\033[0;31m'
    local NC='\033[0m' # No Color

    echo -e "$RED $run_id :  ERROR code: $rv $NC" >&2
    date > $lockfile_failed_fp
    exit $rv
  }
  trap trap_success EXIT  # always run this
  trap trap_err ERR
  trap trap_err INT
}
export -f use_lockfile


function log_initial_msgs() {(
  set -eE
  set -u
  run_id=$1
  echo "Running on hostname: $(hostname)"
  echo "run_id: ${run_id}"
  date

  # print out current configuration
  echo ======================
  echo CURRENT GIT CONFIGURATION:
  echo "git commit: $(git rev-parse HEAD)"
  echo
  echo git status:
  git status
  echo
  echo git diff:
  git --no-pager diff --cached
  git --no-pager diff
  echo
  echo ======================
  echo
  echo
)}
export -f log_initial_msgs


function run_cmd() {
  run_id="$1"
  cmd="$2"
  export -f log_initial_msgs
cat <<EOF | bash
set -eE
set -u
log_initial_msgs $run_id
echo run_id="$run_id" "$cmd"
run_id="$run_id" $cmd
echo job finished
date
EOF
}
export -f run_cmd


function run_cmd_and_log() {
  run_id="$1"
  shift
  cmd="$@"
  lockfile_path=./results/$run_id/lock
  lockfile_runonce=yes

  (
  set -eE
  set -u
  set -o pipefail
  use_lockfile ${lockfile_path} ${lockfile_runonce}
  log_fp="./results/$run_id/`date +%Y%m%dT%H%M%S`.log"
  mkdir -p "$(dirname "$(realpath -m "$log_fp")")"
  run_cmd "$run_id" "$cmd" 2>&1 > $log_fp
  )
}
export -f run_cmd_and_log


# run command and log stdout/stderr
function run() {
  local ri="$1"
  shift
  local cmd="$@"
  run_cmd_and_log $ri $@
}
export -f run


# run jobs with logging of stdout.  useful in conjunction with wait.
#  > fork experiment_name1  some_command
#  > fork experiment_name2  another_command
#  > wait
function fork() {
  (run $@) &
}
export -f fork



function round_robbin_gpu() {
  # distribute `num_tasks` tasks on each of the (locally) available gpus

  # in round robbin fashion.  This implementation is synchronized; a set of
  # `num_tasks` tasks must complete before another set of `num_tasks` tasks
  # starts.

  # NOTE: use `run_gpus` instead if you want one task per gpu, as it doesn't block.

  local num_gpus=$(nvidia-smi pmon -c 1|grep -v \# | awk '{print $1}' | sort -u | wc -l)
  local num_tasks=${1:-$num_gpus}  # how many concurrent tasks per gpu
  local idx=0

  while read -r line0 ; do

    local gpu_idx=$(( $idx % num_gpus ))
    device=cuda:$gpu_idx fork $line0
    local idx=$(( ($idx + 1) % $num_tasks ))
    if [ $idx = 0 ] ; then
      wait # ; sleep 5
    fi
  done
  if [ $idx != 0 ] ; then
    wait # ; sleep 5
  fi
}
export -f round_robbin_gpu


function run_gpus() {
  # Run a set of tasks, one task per gpu, by populating and consuming from a Redis queue.

  # use redis database as a queuing mechanism.  you can specify how to connect to redis with RUN_GPUS_REDIS_CLI 
  local num_tasks_per_gpu="${1:-1}"
  local redis="${RUN_GPUS_REDIS_CLI:-redis-cli -n 1}"
  local num_gpus=$(nvidia-smi pmon -c 1|grep -v \# | awk '{print $1}' | sort -u | wc -l)
  local Q="`mktemp -u -p run_gpus`"

  trap "$(echo $redis DEL "$Q" "$Q/numstarted") > /dev/null" EXIT

  # --> publish to the redis queue
  local maxjobs=0
  while read -r line0 ; do
    $redis LPUSH "$Q" "$line0" >/dev/null
    local maxjobs=$(( $maxjobs + 1 ))
  done
  $redis EXPIRE "$Q" 1209600 >/dev/null # expire queue after two weeks in case trap fails. should make all the rpush events and this expire atomic, but oh well.
  # --> start the consumers
  for gpu_idx in `nvidia-smi pmon -c 1|grep -v \# | awk '{print $1}' | sort -u` ; do
    for i in $(seq $num_tasks_per_gpu) ; do
      consumergpu_redis $gpu_idx "$redis" "$Q" $maxjobs &
  done ; done
  wait
  $redis DEL "$Q" "$Q/numstarted" >/dev/null
}


function consumergpu_redis() {
  local gpu_idx=$1
  local redis="$2"
  local Q="$3"
  local maxjobs=$4

  while /bin/true ; do
    # --> query redis for a job to run
    rv="$($redis --raw <<EOF
MULTI
INCR $Q/numstarted
EXPIRE "$Q" 1209600
EXPIRE "$Q/numstarted" 1209600
RPOPLPUSH $Q $Q
EXEC
EOF
)"
    local cmd="$( echo "$rv" | tail -n 1)"
    local num_started="$( echo "$rv" | tail -n 4| head -n 1)"
    # --> run the job if it hasn't been started previously
    if [ "$num_started" -le "$maxjobs" ] ; then
      device=cuda:$gpu_idx run "$cmd"
    else
      break
    fi
  done
}
export -f run_gpus
