#!/bin/bash
####################################
# clean_folder.sh
# By Nicola Ferralis - 2025.03.31.1
# ##################################

if [ -z "$1" ]; then
      echo "Error: Please provide an argument."
      exit 1
fi

echo $( ls $1 )
for i in $( ls $1 );
do
if [ -d "${i##*.}" ]; then
      echo "Cleaning $i"
      cd "${i##*.}";
      rm -r *.pdf log* *.txt norm* model*
      cd ..
fi
done
