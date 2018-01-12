#!/bin/csh

# Buildbot test script. Execute this script in the top level source dir:
# i.e. /Uintah/trunk/src

# 1. Before running this script the src code must be fully up to date.

# 2. If you are adding new files they must be checked in first.

# 3. There are some other gottas regarding svn ...

# 4. There are different try servers
#   --builder option 

#   where option is one of the following:
#     Linux-Optimize-Test-try
#     Linux-Debug-Test-try
#     Linux-Optimize-GPU-try

#  If no builder is specified all will be used.

# Note thsi script will automatically create a patch via svn (max 640 Mbytes).
# It will contain context lines which are superfluous and make the patch
# bigger than necessary.
buildbot --verbose try --connect=pb --master=uintah-build.chpc.utah.edu:8031 --username=buildbot_try --passwd=try_buildbot --vc=svn --topdir=. --who=`whoami`

# Use this code if your changes are yuuge to create an svn patch with no
# context - still has a max 640 Mbytes. 
#rm -rf buildbot_patch.txt

#svn diff -x --context=0 > buildbot_patch.txt

#buildbot --verbose try --connect=pb --master=uintah-build.chpc.utah.edu:8031 --username=buildbot_try --passwd=try_buildbot --diff=buildbot_patch.txt --who=`whoami` --repository=https://gforge.sci.utah.edu/svn/uintah/trunk/src
