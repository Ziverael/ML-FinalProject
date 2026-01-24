default:
    just --list

tensorboard:
    uv run tensorboard --logdir logs

default_target := './'
default_ignore := ''
format target=default_target ignore=default_ignore:
    #!/usr/bin/env sh
    {{sh_init}}
    format_code {{target}} {{ignore}}

check:
    #!/usr/bin/env sh
    {{sh_init}}
    check_code


alias tb:= tensorboard
alias f:= format

TITLE       := '\033[94m\033[1m'
HIGHLIGHT   := '\033[93m\033[1m'
WARNING     := '\033[91m\033[1m'
DEFAULT     := '\033[0m'

sh_init := "set -e && PROJECT_DIR=$(pwd) && . $PROJECT_DIR/justscripts/main.sh"
