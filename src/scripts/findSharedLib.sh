#!/bin/bash -f
#______________________________________________________________________
#     this script will search through the Uintah/<lib> directory and identify
#     the Uintah libraries that require a specific shared library.
#
#     Usage: 
#       1) cd <libs> directory
#       2) Edit this file and add a searchFor
#       3) execute this script
#______________________________________________________________________

searchFor="/lib64/libgcc_s.so.1"              #<<< change this.

mapfile -t libs < <( find  -name "*.so" )

for lib in  "${libs[@]}"; do
  
  
  if (ldd "$lib" | grep -q  "$searchFor" ); then
    echo "  $lib contains the lib"
    ldd "$lib" | grep "$searchFor"
  fi


done
