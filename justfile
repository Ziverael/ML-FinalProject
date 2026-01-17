default:
    just --list

tensorboard:
    uv run tensorboard --logdir logs

alias tb:= tensorboard