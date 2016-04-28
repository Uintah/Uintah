#!/bin/bash

echo "This script should be called from the root directory"

find src -path src/include -prune -o -not -type d | xargs -n 1 grep -El "\bMPI_[A-Z][_a-z]+\b" | while read file; do
  echo $file
  #Wrap all functions
  sed -i -E "s/\bMPI_([A-Z][_a-z]+)\b/Uintah::MPI::\1/g" $file
  #Restore types
  sed -i -E "s/\bUintah::MPI::Aint\b/MPI_Aint/g" $file
  sed -i -E "s/\bUintah::MPI::Comm\b/MPI_Comm/g" $file
  sed -i -E "s/\bUintah::MPI::Datatype\b/MPI_Datatype/g" $file
  sed -i -E "s/\bUintah::MPI::Op\b/MPI_Op/g" $file
  sed -i -E "s/\bUintah::MPI::Request\b/MPI_Request/g" $file
  sed -i -E "s/\bUintah::MPI::Status\b/MPI_Status/g" $file
done
