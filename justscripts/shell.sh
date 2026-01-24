#!/usr/bin/env sh

# Shell helpers
TITLE="\033[94m\033[1m"
HIGHLIGHT="\033[93m\033[1m"
WARNING="\033[91m\033[1m"
DEFAULT="\033[0m"

echo_default(){
  echo "${DEFAULT}${1}"
}

echo_title(){
  echo "${TITLE}${1}${DEFAULT}"
}

echo_highlight(){
  echo "${HIGHLIGHT}${1}${DEFAULT}"
}

echo_warning(){
  echo "${WARNING}${1}${DEFAULT}"
}

is_arm_architecture(){
  { [ "$(uname -p)" = "arm" ] || [ "${IS_ARM_ARCHITECTURE}" = "true" ]; } && echo true || echo "false"
}

sed_inplace(){
  if [ "$(is_arm_architecture)" = "true" ]
  then
    # MacOS invocation
    sed -i '' "$@"
  else
    # Linux invocation
    sed -i "$@"
  fi
}

get_lowercase_file_extension(){
  FILENAME="${1:?}"
  LOWERCASE_FILENAME="$(echo "${FILENAME}" | tr '[:upper:]' '[:lower:]')"
  LOWERCASE_EXTENSION="${LOWERCASE_FILENAME##*.}"
  [ "${LOWERCASE_FILENAME}" != "${LOWERCASE_EXTENSION}" ] && echo "${LOWERCASE_EXTENSION}"
}
