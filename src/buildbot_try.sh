#! /bin/csh -f

#
# The purpose of this script is to test changes you have made to your local SVN repository
# by shipping those changes to the Uintah buildbot server to run tests against.
#
#
# Before you can run this script, the "buildbot" package must be installed on your computer.
#
# - On Debian (as root) you can run: apt-get install buildbot
#
#
# Other caveats:
#
# - You can't run the try if you have new files in your tree...  The only way around this is
#     to "svn commit" the new files first.
#
#
# This is an example of a successful submission:
#
#   % ./buildbot_try_test.sh
#   2018-04-24 21:52:50-0600 [-] Log opened.
#   2018-04-24 21:52:50-0600 [-] using 'pb' connect method
#   2018-04-24 21:52:52-0600 [-] job created
#   2018-04-24 21:52:52-0600 [-] Starting factory <twisted.spread.pb.PBClientFactory instance at 0x106f2cf38>
#   2018-04-24 21:52:52-0600 [Broker,client] Delivering job; comment= None
#   2018-04-24 21:52:52-0600 [Broker,client] job has been delivered
#   2018-04-24 21:52:52-0600 [Broker,client] not waiting for builds to finish
#   2018-04-24 21:52:52-0600 [-] Stopping factory <twisted.spread.pb.PBClientFactory instance at 0x106f2cf38>
#   2018-04-24 21:52:52-0600 [-] Main loop terminated.

    
#
# Buildbot test script. Execute this script in the top level source dir:
# i.e. /Uintah/trunk/src

# 1. Before running this script the src code must be fully up to date.

# 2. If you are adding new files they must be checked in first.

# 3. There are some other gotchas regarding svn ...

# 4. There are different try servers
#   --builder option 

#   where option is one of the following:
#     Linux-Optimize-Test-try
#     Linux-Debug-Test-try
#     Linux-Optimize-GPU-try

#  If no builder is specified all will be used.

# Note this script will automatically create a patch via svn (max 640 Mbytes).
# It will contain context lines which are superfluous and make the patch
# bigger than necessary.

buildbot --verbose try \
         --connect=pb --master=uintah-build.chpc.utah.edu:8031 \
         --username=buildbot_try --passwd=try_buildbot \
         --vc=svn --topdir=. --who=`whoami`

# Use this code if your changes are yuuge to create an svn patch with no
# context - still has a max 640 Mbytes. 
#rm -rf buildbot_patch.txt

#svn diff -x --context=0 > buildbot_patch.txt

#buildbot --verbose try --connect=pb --master=uintah-build.chpc.utah.edu:8031 --username=buildbot_try --passwd=try_buildbot --diff=buildbot_patch.txt --who=`whoami` --repository=https://gforge.sci.utah.edu/svn/uintah/trunk/src
