#!/usr/bin/env sh
# shellcheck disable=SC2015
# shellcheck disable=SC2120
. justscripts/shell.sh

format_code () {
    TARGET="${1:-./}"
    IGNORE="${2}"
    [ -f "${TARGET}" ] && [ "$(get_lowercase_file_extension "${TARGET}")" != "py" ]\
        && echo "Not a python file. Skipping" && return
    [ -n "${IGNORE}" ] && IGNORE="--ignore ${IGNORE}"
    echo_title "Starting reformat with ruff"
    uv run ruff format "${TARGET}" || ERROR=$?
    # shellcheck disable=SC2086
    uv run ruff check --fix ${IGNORE} "${TARGET}" || ERROR=$?
    show_ruff_hints_if_error_encountered ${ERROR}
}


check_code () {
    echo_title "Starting check with ruff"
    uv run ruff check . || ERROR=$?
    uv run ruff format --check . || ERROR=$?
    show_ruff_hints_if_error_encountered ${ERROR}
    echo_title "Starting check with mypy"
    uv run mypy --incremental --show-error-codes --pretty . || ERROR=$?
    return $ERROR
}

show_ruff_hints_if_error_encountered () {
    ERROR="${1}"
    if [ -n "${ERROR}" ]
    then
        echo_default "Rule details available at https://docs.astral.sh/ruff/rules/"
    fi
}

test_code () {
    TEST_PATH="${1:-./tests}"
    echo_title "Starting tests with pytest"
    OPTS="${3}"
    OPTS="${OPTS} --cov ."
    OPTS="${OPTS} --cov-report html:./coverage/htmlcov"
    OPTS="${OPTS} --cov-report xml:./coverage/coverage.xml"
    OPTS="${OPTS} --cache-clear"
    OPTS="${OPTS} --pyargs ${TEST_PATH}"
    uv  run pytest ${OPTS}
    echo "Coverage report available at $(pwd)/coverage/htmlcov/index.html"
}


open_coverage_report () {
    echo_title "Opening coverage report"
    if [ -f "./coverage/htmlcov/index.html" ]
    then
        firefox ./coverage/htmlcov/index.html
    else
        echo_error "Coverage report not found"
    fi
}

clean_pycached () {
    echo_title "Removing all __pycache__ directories and *.py[cod] files"
    find . -type f -name "*.py[cod]" -delete -or -type d -name "__pycached__" -delete
    echo_default "Done"
}
