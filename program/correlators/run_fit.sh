#!/bin/zsh

# $1 - input file (inis/input_${x}.ini)
# $2 - output file (tag/input_${x}.tag)

python3 ./fit_correlator.py -i $1

touch $2