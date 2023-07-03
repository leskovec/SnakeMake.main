#!/bin/zsh

# shell script that generates the input.ini files for the different tmin, tmax and model

# tmin_min, tmin_max, tmax_min, tmax_max and model are command line arguments

# Parse command line arguments
for arg in "$@"
do
    case $arg in
        --tmin_min=*)
        tmin_min="${arg#*=}"
        shift
        ;;
        --tmin_max=*)
        tmin_max="${arg#*=}"
        shift
        ;;
        --tmax_min=*)
        tmax_min="${arg#*=}"
        shift
        ;;
        --tmax_max=*)
        tmax_max="${arg#*=}"
        shift
        ;;
        *)
        echo "Invalid option $arg"
        exit 1
        ;;
    esac
done

# print opts to screen
echo "generating inis for the followin fit ranges and models"
echo "tmin_min = $tmin_min"
echo "tmin_max = $tmin_max"
echo "tmax_min = $tmax_min"
echo "tmax_max = $tmax_max"

# set x to 0
x=0

for model in "1exp" "2exp"; 
do
    for tmin in $(seq $tmin_min $tmin_max); 
    do
        for tmax in $(seq $tmax_min $tmax_max);
        do
            # replace variables in template
            sed "s|TMIN|${tmin}|g; s|TMAX|${tmax}|g; s|MODEL|${model}|g" input.template.ini > inis/input_${x}.ini

            # increment x by 1
            x=$((x+1))
        done
    done
done
