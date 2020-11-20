#! /bin/csh -f
#
# The purpose of this script is to test changes you have made to your
# local git repository by shipping those changes to the Uintah
# buildbot server to run tests against.
#
# Before you can run this script, the "buildbot" package must be
# installed on your computer.
#
# - On Debian (as root) you can run: apt-get install buildbot
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
#
# 1. Before running this script the src code must be fully up to date.
#
# 2. If you are adding new files they must be checked in first.
#
# 3. To create a patch run "git diff >myPatch"
#
# Usage: buildbot_try.sh   [options]
#             Options:
#              trunk-opt                  Trunk:opt-full-try server
#              trunk-debug/dbg            Trunk:dbg-full-try server
#              trunk-gpu                  Trunk:opt-gpu-try server
#              kokkos-opt                 Kokkos:opt-full-try server
#              all                        run trunk(opt + dbg + gpu) try servers
#              createPatch                run git diff on src/ and submit that patch
#              myPatch      <patchFile>  submit the patchfile to the try servers
#______________________________________________________________________
#

set usage = "/tmp/usage"
cat << EOF > $usage
#______________________________________________________________________
# Usage: buildbot_try.sh   [options]
#             Options:
#              trunk-opt                  Trunk:opt-full-try server
#              trunk-debug/dbg            Trunk:dbg-full-try server
#              trunk-gpu                  Trunk:opt-gpu-try server
#              kokkos-opt                 Kokkos:opt-full-try server
#              all                        run trunk(opt + dbg + gpu) try servers
#              createPatch                run git diff on src/ and submit that patch
#              myPatch      <patchFile>  submit the patchfile to the try servers
#______________________________________________________________________
EOF

# Defaults
set trunkServers = ("Trunk:opt-full-try" "Trunk:dbg-full-try" "Trunk:opt-gpu-try")

set BUILDERS = ""
set CREATE_PATCH = false
set MY_PATCH     = false
set BRANCH       = "master"

# No args so all tests

if ($#argv == 0) then
  foreach server ($trunkServers )
    set BUILDERS = "$BUILDERS --builder=$server"
  end
endif

#__________________________________
# parse inputs
while ( $#argv )
  #echo "($1)"

  # remove punctuation chars and convert to lowercase
  set arg = `echo $1 | tr '[:upper:]' '[:lower:]'`

  switch ($arg:q)
    case createpatch:
      set CREATE_PATCH = true
      shift
      breaksw

    case mypatch:
      set MY_PATCH = true
      set PATCHFILE = $2
      shift; shift
      breaksw

    case trunk-opt:
      set BUILDERS = "$BUILDERS --builder=Trunk:opt-full-try"
      shift
      breaksw
    case trunk-d*b*g:         # debug or dbg
      set BUILDERS = "$BUILDERS --builder=Trunk:dbg-full-try"
      shift
      breaksw

    case trunk-gpu:
      set BUILDERS = "$BUILDERS --builder=Trunk:opt-gpu-try"
      shift
      breaksw

    case kokkos-opt:
      set BUILDERS = "$BUILDERS --builder=Kokkos:opt-full-try"
      set BRANCH = "kokkos_dev"
      shift
      breaksw

    case all:
      foreach server ($trunkServers)
        set BUILDERS = "$BUILDERS --builder=$server"
      end
      shift
      breaksw
    default:
      echo " Error parsing inputs ($1)."
      cat /tmp/usage
      /bin/rm $usage
      echo "   Now exiting"
      exit(1)
      breaksw
  endsw
end

# bulletproofing
if ( $CREATE_PATCH == "false" && $MY_PATCH == "false" ) then
  echo "  ERROR: missing patch option.  Please select one of the following options:"
  echo
  echo "    createPatch               run git diff on src/ and submit that patch"
  echo "    myPatch      <patchFile>  submit the patchfile to the try servers"

  cat $usage
  /bin/rm $usage
  exit 1

endif

/bin/rm $usage
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

  git diff > buildbot_patch.txt

  ls -l buildbot_patch.txt

  set PATCH = "--diff=buildbot_patch.txt --patchlevel=1"

endif
#__________________________________
# use a user created patch

if( $MY_PATCH == "true" ) then
  if( ! -e $PATCHFILE ) then
    echo "  Error:  Could not find the patch file $PATCHFILE"
    exit 1
  endif

 set PATCH = "--diff=$PATCHFILE --patchlevel=1"
endif

#__________________________________

echo "  PATCH ${PATCH}"
echo "  BUILDERS: ${BUILDERS}"

#__________________________________

buildbot --verbose try \
         --connect=pb \
         --master=uintah-build.chpc.utah.edu:8031 \
         --branch=${BRANCH} \
         --repository='https://github.com/Uintah/Uintah.git' \
         --username=buildbot_try \
         --passwd=try_buildbot \
         --vc=git \
         --who=`whoami` \
         ${PATCH} ${BUILDERS}

echo $status

# cleanup
if( $CREATE_PATCH == "true" ) then
  /bin/rm -rf buildbot_patch.txt >& /dev/null
endif

exit
