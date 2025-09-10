#!/bin/bash

pattern='_[0-9]*-par'

for filename in *; do
  if [[ $filename =~ $pattern ]]; then
    new_filename=$(echo "$filename" | sed -E "s/$pattern//")
    mv "$filename" "$new_filename"
    echo "Renamed '$filename' to '$new_filename'"
  fi
done

echo "Renaming process complete."
