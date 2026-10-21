#!/bin/bash
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
#   % ./buildbot_try.sh trunk-opt createPatch
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
#______________________________________________________________________
#

usage()
{
  cat << EOF
#______________________________________________________________________
# Usage: buildbot_try.sh   [options]
#             Options:
#              trunk-opt                  Trunk:optimized tests
#              trunk-debug/dbg            Trunk:debug tests
#              trunk-gpu                  Trunk:optimized gpu tests
#              kokkos-opt                 Kokkos:optimized tests
#              allBuilders                run trunk(opt + dbg + gpu) tests
#              createPatch                run git diff on src/ and submit that patch
#              testComponents             An optional list of components to run tests, space deliminted surrounded my quotes.
#                                         Valid options:
#                                             trunk:  "ARCHES Examples ICE IMPM Models MPMICE MPM UCF Wasatch"
#                                             gpu:    "ARCHES Examples"
#                                             kokkos: "ARCHES Examples"
#              myPatch      <patchFile>   submit the patchfile to the try servers
#______________________________________________________________________
EOF
}

# Defaults
trunkServers=( "Trunk:opt-full" "Trunk:dbg-full" "Trunk:opt-full-gpu" )

BUILDERS=()
CREATE_PATCH=false
MY_PATCH=false
BRANCH="master"
TESTCOMPONENTS=""          # empty: the server uses each builder's default components

# No args so all tests
if [ $# -eq 0 ]; then
  for server in "${trunkServers[@]}"; do
    BUILDERS+=( "--builder=$server" )
  done
fi

#__________________________________
#  check the local buildbot version against the server's
#  ("buildbot try --version" reports only the Twisted version)
#  This requires a gnu:sort

if ! command -v buildbot > /dev/null 2>&1; then
  echo "  ERROR: buildbot is not installed.  On Debian (as root) you can run: apt-get install buildbot"
  exit 1
fi

reqVer="2.0.1"            # buildbot version on uintah-build.chpc.utah.edu
ver=$( buildbot --version | awk '/^Buildbot version:/ {print $3}' )
ver=${ver%%-*}            # drop a Debian package suffix, e.g. 2.0.1-2

oldest=$( printf '%s\n%s\n' "$ver" "$reqVer" | sort --version-sort | head -1 )

if [ -z "$ver" ]; then
  echo "WARNING:  Could not determine the version of buildbot.  Server version: $reqVer"
elif [ "$ver" != "$reqVer" ] && [ "$oldest" == "$ver" ]; then
  echo "WARNING:  The version of buildbot ($ver) is older than the version on the server; this script may not work properly."
  echo "          Server version: $reqVer"
fi

#__________________________________
# parse inputs
while [ $# -gt 0 ]; do
  #echo "($1)"

  # convert to lowercase
  arg=$( echo "$1" | tr '[:upper:]' '[:lower:]' )

  case "$arg" in
    createpatch)
      CREATE_PATCH=true
      shift
      ;;

    mypatch)
      MY_PATCH=true
      PATCHFILE="$2"
      shift; shift
      ;;

    trunk-opt)
      BUILDERS+=( "--builder=Trunk:opt-full" )
      shift
      ;;

    trunk-d*b*g)         # debug or dbg
      BUILDERS+=( "--builder=Trunk:dbg-full" )
      shift
      ;;

    trunk-gpu)
      BUILDERS+=( "--builder=Trunk:opt-full-gpu" )
      shift
      ;;

    kokkos-opt)
      BUILDERS+=( "--builder=Kokkos:opt-full" )
      BRANCH="kokkos_dev"
      shift
      ;;

    allbuilders)
      for server in "${trunkServers[@]}"; do
        BUILDERS+=( "--builder=$server" )
      done
      shift
      ;;

    testcomponents)
      TESTCOMPONENTS="$2"
      shift; shift
      ;;

    *)
      echo " Error parsing inputs ($1)."
      usage
      echo "   Now exiting"
      exit 1
      ;;
  esac
done

# bulletproofing
if [ "$CREATE_PATCH" == "false" ] && [ "$MY_PATCH" == "false" ]; then
  echo "  ERROR: missing patch option.  Please select one of the following options:"
  echo
  echo "    createPatch               run git diff on src/ and submit that patch"
  echo "    myPatch      <patchFile>  submit the patchfile to the try servers"

  usage
  exit 1
fi

#__________________________________
# email address for the build results: your git email, the same address
# buildbot uses for your commits.  A bare login name would be sent to
# <login>@sci.utah.edu.

EMAIL=$( git config user.email )

if [ -z "$EMAIL" ]; then
  echo "  ERROR: no git email address is set.  Set it with:"
  echo "    git config --global user.email you@utah.edu"
  exit 1
fi

#______________________________________________________________________
#
# Note normally git will automatically create a patch.  It will
# contain context lines which are superfluous and make the patch
# bigger than necessary.

# If your changes are huge manually create an svn patch with no
# context - still has a max 640 Mbytes.

PATCH=()

if [ "$CREATE_PATCH" == "true" ]; then
  /bin/rm -rf buildbot_patch.txt > /dev/null 2>&1

  git diff > buildbot_patch.txt

  ls -l buildbot_patch.txt

  PATCH=( "--diff=buildbot_patch.txt" "--patchlevel=1" )
fi

#__________________________________
# use a user created patch

if [ "$MY_PATCH" == "true" ]; then
  if [ ! -e "$PATCHFILE" ]; then
    echo "  Error:  Could not find the patch file $PATCHFILE"
    exit 1
  fi

  PATCH=( "--diff=$PATCHFILE" "--patchlevel=1" )
fi

#__________________________________

echo "  PATCH:      ${PATCH[*]}"
echo "  BUILDERS:   ${BUILDERS[*]}"
echo "  EMAIL:      ${EMAIL}"
if [ -z "$TESTCOMPONENTS" ]; then
  echo "  Test Components: Default components"
else
  echo "  Test Components: ${TESTCOMPONENTS}"
fi


#__________________________________

buildbot --verbose try \
         --connect=pb \
         --master=uintah-build.chpc.utah.edu:8031 \
         --branch="${BRANCH}" \
         --repository='https://github.com/Uintah/Uintah.git' \
         --username=buildbot_try \
         --passwd=try_buildbot \
         --property=test_components="${TESTCOMPONENTS}" \
         --vc=git \
         --who="${EMAIL}" \
         "${PATCH[@]}" "${BUILDERS[@]}"

rc=$?
echo $rc

if [ $rc -eq 0 ]; then
  echo "  Success.  Check http://uintah-build.chpc.utah.edu:8010/#/builders for status"
else
  echo "  Failed"
fi

# cleanup
if [ "$CREATE_PATCH" == "true" ]; then
  /bin/rm -rf buildbot_patch.txt > /dev/null 2>&1
fi

exit $rc
