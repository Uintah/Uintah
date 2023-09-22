#!/bin/bash

#
# The MIT License
#
# Copyright (c) 1997-2023 The University of Utah
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#/
source "$(dirname $0)/bashFunctions"    # pull in common functions

#______________________________________________________________________
# 
#______________________________________________________________________

usage()
{
  echo "DESCRIPTION"
  echo " Paste a series of files containing columns of data together, removing selected columns from files. The first file listed"
  echo " is not modifed, all other files are."  
  echo 
  echo "Usage:"
  echo "  paste -f <FILE> -f <FILE> ..... -r <RANGE> -o <outputFile>"
  echo ""
  echo "Options:"
  echo "      -d, --delimiter DELIM"
  echo -e "              Use DELIM instead of white space as field delimiter\n"
  echo "      -f, --file FILE"
  echo -e "              Files containing columns of values\n"
  echo "      -h, --help"      
  echo -e "              Display usage\n"
  echo "      -r, --range=\"RANGE\""
  echo -e "              Keep all columns in the RANGE.  The first file is not modified\n"
  echo "      -o, --output"
  echo -e "              outputFileName\n"
  echo ""
  echo -e "      The RANGE is made up of one or more ranges separated by commas. 
       Syntax:
       N      N'th column, counted from 1\n
       N-     from N'th column, to end of line\n
       N-M    from N'th to M'th (included) column\n\n" 
  echo "EXAMPLES"
  echo "    Paste two uintah.dat files together removing the first column use:"
  echo "           paste.sh -f 1.dat -f 2.dat -r 2- -o combined.dat"
  echo ""
  echo "    Paste three lineextract output files removing the x,y,z coordinates use:"
  echo "           paste.sh -f temperature.txt -f density.txt -f pressure.txt -r 4- -o combined.txt"
  
  
}

#______________________________________________________________________
#______________________________________________________________________

main()
{
  declare -a orgFiles
  declare -a modifiedFiles
  declare -x tmpDir="$(mktemp -d /tmp/paste-XXX)"
  delim=" "              # delimiter between columns
  flag=0                 # parsing flag
  range=""
  here=$(pwd)
  
  #__________________________________
  # parse arguments
  # ::   value is optional
  # :    something is required
  SHORT="d:,f:,h,o:,r:"
  LONG="delimiter:,file:,help,output:,range:"
  options=$( getopt --name "paste.sh" --options=$SHORT  --longoptions=$LONG -- "$@" )

  if [ $? -ne 0 ] || [ $# -eq 0 ] ; then
    echo "Incorrect option provided"
    usage
    exit 1
  fi

  # set is to preserve white spaces and punctuation
  eval set -- "$options"
  flag=0

  while true ; do
    case "$1" in
      -d|--delimiter)
        shift
        delim="$1"
        ;;
      -h|--help)
        usage
        exit 1
        ;;
      -f|--file)
        shift
        orgFiles+=("$1")
        flag=1
        ;;
      -o|--output)
        shift
        outFile="$1"
        flag=1
        ;;
      -r|--range)
        shift
        range="$1"
        flag=1
        ;;
      --)
        shift
        break
        ;;
    esac
    shift
  done
  
  #__________________________________
  
  if [[ $flag == 0 ]]; then
    usage
    end_die "ERROR:    One of the required inputs was missing"
  fi
  if [[ $range == "" ]]; then
    usage
    end_die "ERROR:    range input missing"
  fi
  
  #__________________________________
  
  echo "__________________________________"
  echo " files:        ${orgFiles[*]}"
  echo " delimiter:    ($delim)"
  echo " range:        ($range)"
  echo " output:       ($outFile)"
  echo "__________________________________"  

  #__________________________________
  #  bulletproofing - do all files exist?
  for f in "${orgFiles[@]}"; do
    if [[ ! -e "${f}" ]]; then
      end_die "  Error:  file ($f) not found"
    fi
  done
  
  
  modifiedFiles[0]=$(basename "${orgFiles[0]}")
  cp "${orgFiles[0]}" "$tmpDir"
  
  #__________________________________
  #  Modify the files, skipping the first file
  
  for f in "${orgFiles[@]:1}"; do
    echo "  -Modifying in $f"
  
    fileName=$(basename "${f}")
    modifiedFiles+=("$fileName")
    
    cut --delimiter="$delim" --fields="$range" "${f}" >& "$tmpDir/$fileName"
  done
  
  #__________________________________
  #  paste the files together
  cd $tmpDir
  touch $outFile
  echo "  -Pasting the files :${modifiedFiles[*]}"

  paste --delimiter="$delim" ${modifiedFiles[@]} >"$outFile"
  mv "$outFile" $here/
  
  # cleanup
  /bin/rm -rf "$tmpDir"
  
}

#______________________________________________________________________
#______________________________________________________________________

main "$@"
