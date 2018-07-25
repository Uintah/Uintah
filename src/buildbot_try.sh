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

#______________________________________________________________________
#
# Have the user choose a try server(s)
set possibleServers =( "Linux-Optimize-Test-try" "Linux-Debug-Test-try" "Linux-Optimize-GPU-try")

set list = ""
foreach comp ( $possibleServers[*] )
  set list="$list $comp - off,"
end
set list = "$list All - off"

set selectedServers = `dialog --stdout --separate-output --checklist "Select the buildbot server(s)" 15 40 15 $list`

# remove quotation marks
set selectedServers = `echo $selectedServers | tr -d '"'`

# bullet proofing
if ( $#selectedServers == "0" ) then                                                                
  echo ""
  echo "Cancel selected... Goodbye."                                                                  
  echo ""
  exit                                                                                    
endif

# define the builders
set BUILDERS = ""
foreach server ($selectedServers )
  set BUILDERS = "$BUILDERS --builder=$server"
end

#______________________________________________________________________
#
# Note this will automatically create a patch via svn 
# It will contain context lines which are superfluous and make the patch
# bigger than necessary.

/bin/rm -rf buildbot_patch.txt >& /dev/null

svn diff > buildbot_patch.txt                
#svn diff -x --context=0 > buildbot_patch.txt   This may be necessary -Todd

buildbot --verbose try \
         --connect=pb \
         --master=uintah-build.chpc.utah.edu:8031 \
         --username=buildbot_try \
         --passwd=try_buildbot \
         --diff=buildbot_patch.txt \
         --vc=svn \
         --topdir=. \
         --who=`whoami` \
         $BUILDERS

# cleanup
/bin/rm -rf buildbot_patch.txt >& /dev/null

exit
