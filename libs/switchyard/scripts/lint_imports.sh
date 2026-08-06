#!/bin/bash

set -eu

errors=0

git --no-pager grep '^from langchain_experimental\.' . && errors=$((errors+1))

exit "$errors"
