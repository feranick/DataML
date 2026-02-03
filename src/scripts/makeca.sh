#!/bin/bash
# Usage: ./makeca.sh <name of mastercsvfile>

csvfile=$1

cd Mat_vs_Mat
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../$csvfile

cd ../../Mat_vs_Perf
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../$csvfile

for i in $(seq 1 8)
do
cd ../Perf$i/full; rm *.csv *.pdf
pwd
CorrAnalysis ../../../$csvfile
cd ../
done

cd ../../Mat_vs_Proc
cd Complete
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../../Kyushu_in_red
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../../

cd ../Mat-Proc_vs_Perf
cd Complete
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../../Kyushu_in_red
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../../

cd ../Perf_vs_Perf
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../$csvfile
cd ../

cd ../Proc_vs_Perf
cd Complete
cd full
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../../$csvfile
cd ../../specific
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../../$csvfile

cd ../../../Kyushu_in_red
pwd
cd full
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../../$csvfile
cd ../../specific
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../../$csvfile

for i in $(seq 1 8)
do
cd ../Perf$i; rm *.csv *.pdf
pwd
CorrAnalysis ../../../../$csvfile
done

cd ../../../../Proc_vs_Proc
pwd
cd Complete
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile

cd ../../Kyushu_in_red
pwd
cd AntiCorr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile
cd ../Corr; rm *.csv *.pdf
CorrAnalysis ../../../$csvfile

