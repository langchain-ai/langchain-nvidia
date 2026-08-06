#!/bin/bash

set -eu

# Initialize a variable to keep track of errors
errors=0

# Make sure not importing from langchain or langchain_experimental.
# Allow langchain.agents and langchain.tools for v1 middleware APIs.
git --no-pager grep "^from langchain\." . | grep -v ":from langchain\.agents" | grep -v ":from langchain\.tools" && errors=$((errors+1))
git --no-pager grep "^from langchain_experimental\." . && errors=$((errors+1))

# Decide on an exit status based on the errors
if [ "$errors" -gt 0 ]; then
    exit 1
else
    exit 0
fi
