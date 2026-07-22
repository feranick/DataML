#!/bin/bash

cleanfile.sh *.pdf $1;
cleanfile.sh *.txt $1;
cleanfile.sh log* $1;
cleanfile.sh model_* $1;
cleanfile.sh *.pkl $1;
cleanmacstuff.sh $1;
