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

usage()
{
  echo "______________________________________________________________________"
  echo 
  echo "DESCRIPTION"
  echo " This script deletes timesteps directories from an uda."
  echo " The timestep directories are moved to a temporary directory that will be deleted"  
  echo " 3 days after the script is executed.  When deleting a large number of timesteps "
  echo " use the -n, --now option to avoid filling /tmp directory"
  echo
  echo "Usage:"
  echo "  1) Edit the uda/index.xml file and remove all timestep directories to delete"
  echo 
  echo "  2) Run:"
  echo "       deleteUdaTimesteps.sh [options] ... [index.xml]"
  echo ""
  echo "Options:"
  echo "    -h, --help     display usage"
  echo "    -n, --now      delete all directories now"
  echo ""
  echo "Dependencies:  at, xmlstarlet"
  echo "______________________________________________________________________"
  exit
}


#______________________________________________________________________
# 
#______________________________________________________________________
main()
{
  #__________________________________
  #  defaults
  TMP="/tmp"                                       # temporary directory
  declare -x rmDirDelay="+3days"                   # Time delay before tmpDir will be deleted
  
  #__________________________________
  # parse arguments
  options=$( getopt --name "deleteUdaTimesteps.sh" --options="h,n"  --longoptions=help,now -- "$@" )

  if [[ $? -ne 0 || $# -eq 0 ]] ; then
    echo "Incorrect option provided"
    usage
    exit 1
  fi
  
  # set is to preserve white spaces and punctuation
  eval set -- "$options"

  while true ; do
    case "$1" in
      -n|--now)
        rmDirDelay="now"
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
  
  index_xml="$1"

  echo "__________________________________"
  echo "  Time delay before timesteps are removed:  $rmDirDelay"
  echo "  index.xml:                                $index_xml"
  echo "__________________________________"
  echo ""

  #__________________________________
  # bulletproofing
  if [ ! -e "${index_xml}" ]; then
    end_die "Couldn't find the index.xml file (${index_xml})."
  fi

  commandExists xmlstarlet
  commandExists atq

  #__________________________________
  #  - find the uda name
  #  - create the tmp directory

  fullpath=$(realpath "$index_xml" | xargs dirname)

  uda=$( echo "$fullpath" | rev | cut --delimit "/" --field 1 | rev )

  tmpDir="$(mktemp -d $TMP/$uda-XXX)"

  if [[ "$?" != "0" ]]; then
    end_die "\n Error: could not create the temporary directory($tmpDir)"
  fi


  #__________________________________
  #  Parse the index.xml file for the timestep directories

  keepDirs=$(xmlstarlet sel --text --template --value-of "/Uintah_DataArchive/timesteps/timestep/@href" "$index_xml" ) 

  echo $keepDirs | xargs dirname > "$tmpDir/keepDirs"

  #__________________________________
  # find all timestep directories in the uda

  cd "$fullpath"

  find ./t* -maxdepth 0 -type d  -exec basename {} \; > "$tmpDir/allDirs"

  sort --version-sort  "$tmpDir/allDirs" --output "$tmpDir/allDirs"

  if [[ "$?" != "0" ]]; then
    end_die "\n Error: did not find any timestep directories in $uda"
  fi

  #__________________________________
  # find the directories to delete.  This is the difference of allDirs vs keepDirs

  declare -a deleteDirs=( $( diff  --changed-group-format='%>' --unchanged-group-format='' "$tmpDir/keepDirs"  "$tmpDir/allDirs" | tr  "\n" " ") )


  if [[ ${#deleteDirs[@]} -eq 0 ]]; then
    end_die "Error:  No directories will be deleted.  Make sure you've edited the index.xml file and remove timestep directories."
  fi


  #__________________________________
  #   Move the directories to the tmpdir or delete them
  if [[ "$rmDirDelay" == "now" ]]; then
    echo "Proceed with deleting the timestep directories immediately?" 
  else
    echo "Proceed with moving the timestep directories to $tmpDir and then delete that directory $rmDirDelay?"
  fi

  for f in "${deleteDirs[@]}"; do
    
    if [[ "$rmDirDelay" == "now" ]]; then
      echo "   /bin/rm -rf $f"
    else
      echo "   mv -r $f $tmpDir"
    fi
  done


  while true; do
      read -p "Y/N  " ans
      case $ans in

        [Yy]* ) 
          
          if [[ "$rmDirDelay" != "now" ]]; then
            for f in "${deleteDirs[@]}"; do
              echo "   mv -r $f $tmpDir"
              mv -r "$f" "$tmpDir"
            done
          fi
          
          if [[ "$rmDirDelay" == "now" ]]; then
            for f in "${deleteDirs[@]}"; do
              echo "   /bin/rm -rf $f"
              /bin/rm -rf "$f"
            done
          fi          
          
          cleanUp "${tmpDir}" ${rmDirDelay}
          break;;

        [Nn]* ) 
          exit;;
        * ) echo "Please answer Y or N.";;
      esac
  done

  if [[ "$rmDirDelay" != "now" ]]; then
    echo "If you've made a mistake you can simply move the directories from $tmpDir/ back to $uda/"
  fi
  exit
}
#______________________________________________________________________
#______________________________________________________________________

main "$@"

