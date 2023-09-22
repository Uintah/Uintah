#!/bin/bash -f

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
#______________________________________________________________________

source "$(dirname $0)/bashFunctions"    # pull in common functions

usage()
{
  echo ""
  echo " Usage:"
  echo "    mkdir masterUda        MANDATORY"
  echo "    makeMasterUda_index.sh [options] uda.000  uda.001 .... uda.00N"
  echo ""
  echo " Options:"
  echo "     -f | --fullpath          Output the full path to the uda.  The default"
  echo "                              is to output the user's input path."
  echo "     -h | --help"
  exit
}

#______________________________________________________________________
#
#______________________________________________________________________
main()
{
  #__________________________________
  #  defaults
  tmpDir=$(mktemp -d /tmp/makeMasterUda-XXX)    # temporary directory
  cmdOption=""                                  # output the full path to each uda
  masterUda="masterUda"                         # directoryName for masterUda
  rootPath=$( realpath "$0" | xargs dirname )   # define the path to the scripts directory
  export PATH="$PATH:$rootPath"

  #__________________________________
  #  Parse the inputs
  options=$( getopt --name "makeMasterUda" --options="h,f" --longoptions=help,fullpath -- "$@" )

  if [[ $? -ne 0 || $# -eq 0 ]]; then
    echo "Terminating..." >/dev/stderr
    usage
    exit 1
  fi

  eval set -- "$options"

  while true ; do
    case "$1" in
      -f|--fullpath)
        echo "output fullpath"
        cmdOption="-f"
        ;;
      -h|--help)
        usage
        ;;
      --)
        shift
        break
        ;;
    esac
    shift
  done

  declare -a udas
  udas=($@)

  #__________________________________
  # bulletproofing
  commandExists makeCombinedIndex.sh

  if [[ ! -e "$masterUda" ]]; then
    end_die "Couldn't find the $masterUda directory."
  fi

  echo ""
  echo "---------------------------------------"
  for uda in "${udas[@]}"; do
    echo "Passing $uda through bulletproofing section"

    # does each uda/index.xml exist
    if [[ ! -e "$uda/index.xml" ]]; then

      echo "Working on $uda "
      echo "______________________________________________________________"
      echo "ERROR: makeMasterUda: can't find the file $uda/index.xml"
      echo "                   Do you want to continue"
      echo "                             Y or N"
      echo "______________________________________________________________"
      read -p "Y/N  " ans

      if [[ $ans == "n" || $ans == "N" ]]; then
        exit
      fi
    fi
  done

  #__________________________________
  # Look for difference in the variable lists between the udas
  # Warn the users if one is detected
  echo ""
  echo "---------------------------------------"
  echo "Looking for differences in the variable lists"

  io_type="UDA"

  # copy the variables section of each index.xml to the tmpDir for further analysis below
  for uda in "${udas[@]}"; do

    here=$( basename "$uda" )

    grep variable "$uda/index.xml" > "$tmpDir/$here"

    # check if io type is uda or PIDX
    if [[ $(grep -c "<outputFormat>PIDX</outputFormat>" "$uda/index.xml") == 1 ]]; then
      io_type="PIDX"
    fi
  done

  echo $io_type

  #__________________________________
  # look for differences in the variable list of the index.xml files
  # and warn the user
  udas_tmp=( ${udas[@]} )

  for uda1 in "${udas_tmp[@]}"; do
    udas_tmp=( ${udas_tmp[@]/$uda1} )  # remove $uda1 from the tmp array

    for uda2 in "${udas_tmp[@]}"; do

      X=$( basename "${uda1}" )
      Y=$( basename "${uda2}" )

      ans1=$( grep -c variables "$tmpDir//$X" )
      ans2=$( grep -c variables "$tmpDir//$Y" )

      if [[ $ans1 == "2" && $ans2 == "2" ]]; then
        diff -B "$tmpDir/$X" "$tmpDir//$Y" >& /dev/null

        if [[ $? -ne 0 ]]; then
          echo ""
          echo "  WARNING:  A difference in the variable list was detected between $X/index.xml and $Y/index.xml"
          echo ""
          sdiff -s -w 170 "$tmpDir/$X" "$tmpDir/$Y"
        fi
      fi
    done
  done

  #__________________________________
  #  copy uda[0]/input.xml and input.xml.orig to masterUda
  echo ""
  echo "__________________________________"
  echo "Copying ${udas[0]}/input.xml and input.xml.orig to $masterUda"

  if [[ ! -e "$masterUda/input.xml" ]]; then
    cp "${udas[0]}/input.xml" "$masterUda"
  fi

  if [[ ! -e $masterUda/input.xml.orig ]]; then
    cp "${udas[0]}/input.xml.orig" "$masterUda"
  fi

  #__________________________________
  # copy uda[0]/index.xml to masterUda
  # remove all the timestep data
  echo ""
  echo "---------------------------------------"
  echo "Creating the base index file from ${udas[0]}"

  #             copy index.xml and remove all timestep entries
  sed /"timestep "/d "${udas[0]}/index.xml" > "$masterUda/index.xml"

  #             remove   </timesteps> to </Uintah_DataArchive>
  sed -i  /"\/timesteps"/,/"\/Uintah_DataArchive"/d "$masterUda/index.xml"

  #             Add the updated <timstep> entries to index.xml
  makeCombinedIndex.sh "$cmdOption" "${udas[@]}" >> "$masterUda/index.xml"

  # generate all gidx
  if [[ "$io_type" == "PIDX" ]]; then
    rm "*.gidx"
    makeCombinedGIDX.sh "${udas[@]}"
  fi

  # put the closing xml tags back
  echo  "  </timesteps>"        >> "$masterUda/index.xml"
  echo  "</Uintah_DataArchive>" >> "$masterUda/index.xml"


 #__________________________________
 # cleanup
 /bin/rm -rf "$tmpDir"
exit
}
#______________________________________________________________________
#______________________________________________________________________

main "$@"

