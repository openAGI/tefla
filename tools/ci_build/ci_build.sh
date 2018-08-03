#!/usr/bin/env bash
# Copyright 2018 The Tefla Authors. All Rights Reserved.

SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
ROOT_DIR=$( cd "$SCRIPT_DIR/../.." && pwd -P )

run_unit_tests()
{
  echo "#########################################"
  echo "Running unit tests"
  pytest -v "${ROOT_DIR}/tests/"
  echo "Done running tests"
  echo "#########################################"
  if [[$? ! =0]]; then
    return 1
  else
    return 0
  fi
}

run_sanity_tests()
{
  echo "#########################################"
  echo "Running unit tests"
  bash "${SCRIPT_DIR}/ci_sanity.sh"
  echo "Done running tests"
  echo "#########################################"
  if [[$? ! =0]]; then
    return 1
  else
    return 0
  fi
}

TESTS_STEPS=("run_unit_tests PYTHON3" "run_sanity_tests")
TESTS_STEPS_DESC=("Python3 Unit tests" "Sanity tests")
# TESTS_STEPS=("run_sanity_tests")
# TESTS_STEPS_DESC=("Sanity tests")

INCREMENTAL_FLAG=""
DEFAULT_BAZEL_CONFIGS="--config=hdfs --config=gcp"

FAIL_COUNTER=0
PASS_COUNTER=0
STEP_EXIT_CODES=()

# Execute all the sanity build steps
COUNTER=0
while [[ ${COUNTER} -lt "${#TESTS_STEPS[@]}" ]]; do
  INDEX=COUNTER
  ((INDEX++))

  echo ""
  echo "=== Unit tests check step ${INDEX} of ${#TESTS_STEPS[@]}: "\
"${TESTS_STEPS[COUNTER]} (${TESTS_STEPS_DESC[COUNTER]}) ==="
  echo ""

  # subshell: don't leak variables or changes of working directory
  (
  ${TESTS_STEPS[COUNTER]} ${INCREMENTAL_FLAG}
  )
  RESULT=$?

  if [[ ${RESULT} != "0" ]]; then
    ((FAIL_COUNTER++))
  else
    ((PASS_COUNTER++))
  fi

  STEP_EXIT_CODES+=(${RESULT})

  echo ""
  ((COUNTER++))
done

# Print summary of build results
COUNTER=0
echo "==== Summary of sanity check results ===="
while [[ ${COUNTER} -lt "${#TESTS_STEPS[@]}" ]]; do
  INDEX=COUNTER
  ((INDEX++))

  echo "${INDEX}. ${TESTS_STEPS[COUNTER]}: ${TESTS_STEPS_DESC[COUNTER]}"
  if [[ ${STEP_EXIT_CODES[COUNTER]} == "0" ]]; then
    printf "  ${COLOR_GREEN}PASS${COLOR_NC}\n"
  else
    printf "  ${COLOR_RED}FAIL${COLOR_NC}\n"
  fi

  ((COUNTER++))
done

echo
echo "${FAIL_COUNTER} failed; ${PASS_COUNTER} passed."

echo
if [[ ${FAIL_COUNTER} == "0" ]]; then
  printf "Unit tests checks ${COLOR_GREEN}PASSED${COLOR_NC}\n"
else
  printf "Unit tests checks ${COLOR_RED}FAILED${COLOR_NC}\n"
  exit 1
fi
