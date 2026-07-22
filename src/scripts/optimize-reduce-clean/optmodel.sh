#!/bin/bash

PYTHONUNBUFFERED=1 DataML_DF -o $1 $2 $3 > tmp.txt 2>&1;
head -n 100 tmp.txt;
cat DataML_DF.ini > log.txt;
tail -n 60 tmp.txt >> log.txt;
DataML_DF -t $2 $3 >> log.txt;
pwd >> log.txt;
rm tmp.txt
cat log.txt
