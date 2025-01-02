#!/bin/bash

for i in 0 1 2 3 4; do
  start=$(date +%s)
  python algo.py "$i"
  end=$(date +%s)
  echo "========================================================"
  echo "Completed: python algo.py $i in $((end - start)) seconds"
  echo "========================================================"
done
