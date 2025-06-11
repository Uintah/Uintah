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
#           <Module name="lineExtract">
#             <material>Atmosphere</material>
#             <samplingFrequency> 1e10 </samplingFrequency>
#             <timeStart>          0   </timeStart>
#             <timeStop>          100  </timeStop>
#
#             <Variables>
#               <analyze label="press_CC" matl="0"/>
#               <analyze label="rho_CC"/>
#               <analyze label="temp_CC"/>
#               <analyze label="delP_Dilatate"/>
#             </Variables>
#
#             <lines>
#               <line name="X_line">
#                      <startingPt>  [0.0, 0.5, 0.5]   </startingPt>
#                      <endingPt>    [1.0, 0.5, 0.5]   </endingPt>
#                      <stepSize> 0.1 </stepSize>
#                </line>
#            </Module>
#       </DataAnalysis>
#
#   The script glues together all data files in a given line for each timestep.
#   It assumes that following directory structure exists:
#
#    X_line/                        < Extracted line
#    |-- L-0                        <level
#    |   |-- i0_j20_k20             < cell indice
#    |   |-- i12_j20_k20
#    |   |-- i16_j20_k20
#    |   |-- i20_j20_k20
#    |   |-- i24_j20_k20
#    |   |-- i28_j20_k20
#    |   |-- i32_j20_k20
#    |   |-- i36_j20_k20
#    |   |-- i4_j20_k20
#    |   `-- i8_j20_k20
#    `-- L-1                        < optional level
#        |-- i20_j40_k40
#        |-- i28_j40_k40
#        |-- i30_j40_k40
#        |-- i38_j40_k40
#        |-- i40_j40_k40
#        |-- i48_j40_k40
#        |-- i50_j40_k40
#        `-- i58_j40_k40
#
#
#  Each file has the format of
#
#   X_CC           Y_CC             Z_CC             Time [s]        press_CC_0      rho_CC_0        temp_CC_0       delP_Dilatate_0
#   1.250000E-02     5.125000E-01     5.125000E-01     2.701815E-05    1.013250E+05    1.789991E+00    3.000000E+02    -0.000000E+00
#   1.250000E-02     5.125000E-01     5.125000E-01     5.403629E-05    1.013250E+05    1.789991E+00    3.000000E+02    -0.000000E+00
#   1.250000E-02     5.125000E-01     5.125000E-01     8.105444E-05    1.013250E+05    1.789991E+00    3.000000E+02    -0.000000E+00
#   <snip>
#
#   This script generates a new directory inside each level
#   #______________________________________________________________________
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
#  Where X_line/L-0/timesteps/t1.080726E-4
# X_CC           Y_CC             Z_CC             Time [s]        press_CC_0      rho_CC_0        temp_CC_0       delP_Dilatate_0 
#  1.250000E-02     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    1.789991E+00    3.000000E+02    -0.000000E+00   
#  1.125000E-01     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    1.789991E+00    3.000000E+02    -0.000000E+00   
#  2.125000E-01     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    1.789991E+00    3.000000E+02    5.310202E-12    
#  3.125000E-01     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    1.789991E+00    3.000000E+02    -0.000000E+00   
#  4.125000E-01     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    7.711993E-01    3.000000E+02    -3.625098E-12   
#  5.125000E-01     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    1.789991E-12    3.000000E+02    -0.000000E+00   
#  6.125000E-01     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    1.018791E+00    3.000000E+02    -7.250196E-12   
#  7.125000E-01     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    1.789991E+00    3.000000E+02    1.770067E-12    
#  8.125000E-01     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    1.789991E+00    3.000000E+02    -0.000000E+00   
#  9.125000E-01     5.125000E-01     5.125000E-01     1.080726E-04    1.013250E+05    1.789991E+00    3.000000E+02    -0.000000E+00 
# 
# 
#______________________________________________________________________

usage()
{
  echo "______________________________________________________________________"
  echo
  echo "DESCRIPTION"
  echo " This scripts creates a series of files, one for each timestep output"
  echo " and each file contains the columns"
  echo "#   X_CC           Y_CC             Z_CC             Time [s]         <Var1>          <Var2>        ...."
  echo "   1.250000E-02     5.125000E-01     5.125000E-01     2.701815E-05    1.013250E+05    1.789991E+00  ...."
  echo "   1.250000E-02     5.125000E-01     5.125000E-01     5.403629E-05    1.013250E+05    1.789991E+00  ...."
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
    head -1 "${cellFiles[1]}" > "$header"

    #   find all the timesteps in the cell file
    mapfile -t timesteps < <(sed  -e "1d" "${cellFiles[1]}" | \
                             tr --squeeze-repeats ' '| \
                             cut -d ' ' -f4 )

    #__________________________________
    #     Loop over the timesteps and 
    #     create a file with a line of data
    for timestep in "${timesteps[@]}"; do

      ts="timesteps/t$timestep"
      cp "$header" "$ts"

      grep -h "$timestep"  "${cellFiles[@]}" >> "$ts"
      echo "    Working on timestep $ts"

      # sort the data by $sortOrder
      sort "${sortOrder[@]}" "$ts" -o "$ts"

    done
    #     cleanup
    /bin/rm "$header"

    cd ..

  done
}
#______________________________________________________________________
#______________________________________________________________________

main "$@"
