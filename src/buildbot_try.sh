#! /bin/csh -f
#
# The purpose of this script is to test changes you have made to your
# local SVN repository by shipping those changes to the Uintah
# buildbot server to run tests against.
#
#
# Before you can run this script, the "buildbot" package must be
# installed on your computer.

#
# - On Debian (as root) you can run: apt-get install buildbot
#
#
# Other caveats:
#

# - You can't run the try if you have new files in your tree...  The
#     only way around this is to "svn commit" the new files first.
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

# If you want the the gui to be the default uncomment out this line:
#set argv = "gui"

#______________________________________________________________________
#
# Have the user choose a try server(s)
set possibleServers = ("opt-full-try" "dbg-full-try" "opt-gpu-try")

set BUILDERS = ""
set CREATE_PATCH = false

# No args so all tests
if ($#argv == 0) then
  foreach server ($possibleServers )
    set BUILDERS = "$BUILDERS --builder=$server"
  end
else
  foreach arg ( $argv )
    if( $arg == "patch" ) then
      set CREATE_PATCH = true
    else if( $arg == "all" ) then
      foreach server ($possibleServers )
        set BUILDERS = "$BUILDERS --builder=$server"
      end
    else if( $arg == "opt" ) then 
        set BUILDERS = "$BUILDERS --builder=opt-full-try"
    else if( $arg == "debug" ) then
        set BUILDERS = "$BUILDERS --builder=dbg-full-try"
    else if( $arg == "gpu" ) then
        set BUILDERS = "$BUILDERS --builder=opt-gpu-try"
    else # gui
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
      foreach server ($selectedServers )
        set BUILDERS = "$BUILDERS --builder=$server"
      end
    endif
  end
endif

#______________________________________________________________________
#
# Note normally svn will automatically create a patch.  It will
# contain context lines which are superfluous and make the patch
# bigger than necessary.

# If your changes are huge manually create an svn patch with no
# context - still has a max 640 Mbytes.

set PATCH = ""

if( $CREATE_PATCH == "true" ) then
  /bin/rm -rf buildbot_patch.txt >& /dev/null

  svn diff -x --context=0 > buildbot_patch.txt

  ls -l buildbot_patch.txt
  
  set PATCH = "--diff=buildbot_patch.txt --repository=https://gforge.sci.utah.edu/svn/uintah/trunk/src" 
endif

buildbot --verbose try \
         --connect=pb \
         --master=uintah-build.chpc.utah.edu:8031 \
         --username=buildbot_try \
         --passwd=try_buildbot \
         --vc=svn \
         --topdir=. \
         --who=`whoami` \
         $PATCH $BUILDERS

# cleanup
if( $CREATE_PATCH == "true" ) then
  /bin/rm -rf buildbot_patch.txt >& /dev/null
endif

exit
