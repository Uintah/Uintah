#!/bin/sh
#______________________________________________________________________
#  This script is called by the Makefile and updates the git log information
#
#  in <opt/dbg>/include/git_info.h
#
#  The variables (GIT_BRANCH, GIT_DATE AND GIT_HASH) are then output by sus.cc
#______________________________________________________________________

SRCTOP_ABS=$1
OBJTOP_ABS=$2

# copy template file if it doesn't exist
if [ ! -f ${OBJTOP_ABS}/include/git_info.h ]; then
  cp ${SRCTOP_ABS}/include/git_info.h.template  ${OBJTOP_ABS}/include/git_info.h
fi

# execute git log
git_log=$( git -C ${SRCTOP_ABS} log -1 --format="%ad|%d|%H"   2> /dev/null )

# if the command was successful
if [ $? -eq 0 ]; then

   hash=$(   echo ${git_log} | cut -f 3 -d \| )
   branch=$( echo ${git_log} | cut -f 2 -d \| )
   date=$(   echo ${git_log} | cut -f 1 -d \| )

   grep ${hash} ${OBJTOP_ABS}/include/git_info.h > /dev/null 2>&1
   
   # if the hash changed create a new git_info.h file
   if [ $? -ne 0 ]; then
     echo "Updating git log information in include/git_info.h"
     sed -e "s%unknown_hash%$hash%"         \
         -e "s%unknown_date%$date%"         \
         -e "s%unknown_branch%$branch%"     \
         ${SRCTOP_ABS}/include/git_info.h.template > ${OBJTOP_ABS}/include/git_info.h

     echo "  date:   $date "
     echo "  hash:   $hash "
     echo "  branch: $branch "
   fi
else
   echo "______________________________________________________________________"
   echo "Info: git log command failed, no updates to include/git_info.h "
   echo "         The output from sus may not have accurate git information"
   echo "______________________________________________________________________"
fi
exit
