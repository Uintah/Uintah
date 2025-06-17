#!/bin/bash
helpersPath=$( dirname "${BASH_SOURCE[0]}")
source "${helpersPath}"/bashFunctions
#______________________________________________________________________
#
# The MIT License
#
# Copyright (c) 1997-2025 The University of Utah
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

#______________________________________________________________________
#  combineLineExtractData:
#
#   This script is used in conjunction with on-the-fly data analysis module
#       <DataAnalysis>
#               <Module name="lineExtract">
#
#                 <material>           atmosphere  </material>
#                 <samplingFrequency>  1000        </samplingFrequency>
#                 <timeStart>          0.000       </timeStart>
#                 <timeStop>           10000       </timeStop>
#
#                 <Variables>
#                   <analyze label="g.mass"     matl="0"/>
#                   <analyze label="g.velocity" matl="0"/>
#                   <analyze label="g.stressFS" matl="0"/>
#                 </Variables>
#
#                 <lines>
#                   <line name="X_line">
#                      <startingPt>  [0.0, 1.25, 1.25]   </startingPt>
#                      <endingPt>    [2.5, 1.25, 1.25]   </endingPt>
#                      <stepSize> 0.1 </stepSize>
#                    </line>
#                    <snip>
#                   </lines>
#                 </Module>
#         </DataAnalysis>
#
#   The script glues together all data files in a given line for each timestep.
#   It assumes that following directory structure exists:
#
#    X_line/
#    `-- L-0
#        |-- i0_j25_k25
#        |-- i10_j25_k25
#        |-- i12_j25_k25
#        |-- i14_j25_k25
#        |-- i16_j25_k25
#        |-- i18_j25_k25
#        |-- i20_j25_k25
#        |-- i22_j25_k25
#        |-- i24_j25_k25
#        |-- i25_j25_k25
#        |-- i27_j25_k25

#
#  Each file has the format of
#
#
# X_NC           Y_NC             Z_NC             Timestep  Time [s]        g.mass_0        g.velocity_0.x   g.velocity_0.y   g.velocity_0.z  g.stressFS_0(0,0)    g.stressFS_0(0,1)   <snip>
#  0.000000E+00     1.250000E+00     1.250000E+00     1       2.101867E-05    1.000000E-200   0.000000E+00     0.000000E+00     0.000000E+00    0.000000E+00         0.000000E+00
#  0.000000E+00     1.250000E+00     1.250000E+00     49      1.029915E-03    1.000000E-200   0.000000E+00     0.000000E+00     0.000000E+00    0.000000E+00         0.000000E+00
#  0.000000E+00     1.250000E+00     1.250000E+00     97      2.038811E-03    1.000000E-200   0.000000E+00     0.000000E+00     0.000000E+00    0.000000E+00         0.000000E+00
#   <snip>
#
#   This script generates a new directory (timesteps) inside each level
#   #______________________________________________________________________
# X_line/
# `-- L-0
#     `-- timesteps
#         |-- t00001
#         |-- t00049
#         `-- t00097
#
#

#      X_line/
#      |-- L-0
#      |   `-- timesteps
#      |       |-- t1.080726E-04
#      |       |-- t1.350907E-04
#      |       |-- t1.621089E-04
#      |       |-- t1.891270E-04
#      |           <snip>
#      `-- L-1
#          `-- timesteps
#              |-- t1.080726E-04
#              |-- t1.350907E-04
#              |-- t1.621089E-04
#              |-- t1.891270E-04
#              <snip>
#
#  Where X_line/L-0/timesteps/t00097
#  X_NC           Y_NC             Z_NC             Timestep  Time [s]        g.mass_0        g.velocity_0.x   g.velocity_0.y   g.velocity_0.z  g.stressFS_0(0,0)    g.stressFS_0(0,1)  <snip>
# 0.000000E+00     1.250000E+00     1.250000E+00     97        2.038811E-03    1.000000E-200   0.000000E+00     0.000000E+00     0.000000E+00    0.000000E+00         0.000000E+00
# 1.000000E-01     1.250000E+00     1.250000E+00     97        2.038811E-03    1.000000E-200   0.000000E+00     0.000000E+00     0.000000E+00    0.000000E+00         0.000000E+00
# 2.000000E-01     1.250000E+00     1.250000E+00     97        2.038811E-03    1.000000E-200   0.000000E+00     0.000000E+00     0.000000E+00    0.000000E+00         0.000000E+00
# 3.000000E-01     1.250000E+00     1.250000E+00     97        2.038811E-03    1.000000E-200   0.000000E+00     0.000000E+00     0.000000E+00    0.000000E+00         0.000000E+00

#______________________________________________________________________

usage()
{
  echo "______________________________________________________________________"
  echo
  echo "DESCRIPTION"
  echo " This scripts creates a series of files, one for each timestep output"
  echo " and each file contains the columns"
  echo "#   X_CC           Y_CC             Z_CC           Timestep    Time [s]         <Var1>          <Var2>        ...."
  echo "   1.250000E-02     5.125000E-01     5.125000E-01  1           2.701815E-05    1.013250E+05    1.789991E+00  ...."
  echo "   1.250000E-02     5.125000E-01     5.125000E-01  97          5.403629E-05    1.013250E+05    1.789991E+00  ...."
  echo ""
  echo " These files can then be used to plot profiles of the quantities of interest."
  echo ""
  echo "Usage:"
  echo "         combineLineExtractData.sh --dir <X, Y, Z> <path to line extract directory>"
  echo "Options:"
  echo "    -h, --help          display usage"
  echo "    -d, --dir    <X,Y,Z> Principal direction of the line"
  echo "                         This determines the order of sorting the columns"
  echo "______________________________________________________________________"
  exit
}

#______________________________________________________________________
#   Assumption:
#______________________________________________________________________
main()
{
  #  Defaults
  declare -i width=5     #"used for filename format t<width> timestep"

  #__________________________________
  # parse arguments
  options=$( getopt --name "combineLineExtractData.sh" --options="h,d:"  --longoptions=help,dir: -- "$@" )


  if [[ $? -ne 0 || $# -eq 0 ]] ; then
    echo "Incorrect option provided"
    usage
    exit 1
  fi

  # set is to preserve white spaces and punctuation
  eval set -- "$options"

  while true ; do
    case "$1" in
      -d|--dir)
        shift
        dir="${1^^}"        # convert to upper case
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

  #__________________________________
  #  create the sorting order which depends on the line direction
  case "$dir" in
    X)
      sortOrder=(-k1,1g -k2,2g -k3,3g)
      echo "  The cell coordinates are sorted by x then y and z"
      ;;
    Y)
      sortOrder=(-k2,2g -k1,1g -k3,3g)
      echo "  The cell coordinates are sorted by y then x and z"
      ;;
    Z)
      sortOrder=(-k3,3g -k2,2g -k1,1g)
      echo "  The cell coordinates are sorted by z then y and x"
      ;;
     *)
      echo "ERROR: Valid option for -d|--dir is 'X', 'Y' or 'Z' "
      echo "       exiting..."
      exit
      ;;
  esac


  here=$( pathExists "$1" )

  cd "$here"
  #__________________________________
  #       loop over levels
  mapfile -t  Levels < <( find -type d -name "L-*" )

  for L in "${Levels[@]}"; do
    cd "$L"

    echo "  Working on $L"
    mkdir timesteps

    #   find all the cell files on this level
    mapfile -t  cellFiles < <( find -name "i*" )

    #    extract the header
    header="timesteps/header"
    grep ^# "${cellFiles[1]}" > "$header"


    #   create an array of all the timesteps in the cell file
    mapfile -t timesteps < <(sed  '/^#/d' "${cellFiles[1]}" | \
                             tr --squeeze-repeats ' '| \
                             cut -d ' ' -f4 )

    #   create an array of all the physical times in the cell file
    mapfile -t phyTimes < <(sed  '/^#/d' "${cellFiles[1]}" | \
                             tr --squeeze-repeats ' '| \
                             cut -d ' ' -f5 )

    #__________________________________
    #     Loop over the timesteps and
    #     create a file with a line of data
    for ((t=0; t < "${#timesteps[@]}"; t++)); do

      printf -v padded_ts "timesteps/t%0${width}d" "${timesteps[t]}"


      cp "$header" "$padded_ts"

      #       search all the files for the physical time, don't use the timestep:
      grep -h "${phyTimes[t]}"  "${cellFiles[@]}" >> "$padded_ts"
      echo "    Working on timestep $padded_ts"

      #       sort the data by $sortOrder
      sort "${sortOrder[@]}" "$padded_ts" -o "$padded_ts"

    done
    #     cleanup
    /bin/rm "$header"

    cd ..

  done
}
#______________________________________________________________________
#______________________________________________________________________

main "$@"
