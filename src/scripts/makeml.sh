#!/bin/bash
# Usage: ./makeml.sh <name of mastercsvfile>

csvfile=$1
cd ./full

for i in $(seq 1 8)
do
  cd Perf$i
  pwd
  cd MatPar
  pwd
  rm *_test.txt *_train.txt
  DataML_Maker ../../../$csvfile
  cd ../MatProcPar
  pwd
  rm *_test.txt *_train.txt
  DataML_Maker ../../../$csvfile
  cd ../ProcPar
  pwd
  rm *_test.txt *_train.txt
  DataML_Maker ../../../$csvfile
  cd ../..
done
