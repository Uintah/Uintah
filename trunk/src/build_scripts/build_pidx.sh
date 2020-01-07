#! /bin/bash

#
# This script is used by Uintah's configure to build the PIDX thirdparty library
#
# $1 - location to build the 3P
# $2 - whether Uintah is being built in debug mode or not...
#

# Note: Change PIDX_TAG to "master" if you want to grab the latest trunk.
# this was "v0.9.2", changing it temporarly to master (see TODO note below when switching from master to release version)
PIDX_TAG="master"

PIDX_DIR=$1

############################################################################

show_error()
{
    echo ""
    echo "An error occurred in the buid_pidx.sh script:"
    echo ""
    echo "  The error was from this line: $@"
    echo ""
    exit 1
}

############################################################################
# Run() runs the command sent in, and tests to make sure it succeeded.

run()
{
    echo "$@"
    eval $@
    if test $? != 0; then
        show_error "$@"
    fi
}

############################################################################

echo   ""
echo   "------------------------------------------------------------------"
echo   "Building PIDX Library..."
echo   ""
echo   ""
echo   "    Using Cmake: "`which cmake`
echo   ""
echo   "------------------------------------------------------------------"

if  [ ! -d "$PIDX_DIR" ]; then
  run "git clone --branch ${PIDX_TAG} --depth=1 https://github.com/sci-visus/PIDX.git $PIDX_DIR"
fi
run "cd $PIDX_DIR"

if  [ ! -d "./build" ]; then
  run "mkdir build"
fi

run "git config --unset-all remote.origin.fetch"
run "git config --add remote.origin.fetch +refs/tags/${PIDX_TAG}:refs/tags/${PIDX_TAG}"
# TODO change ${PIDX_TAG} to "tags/${PIDX_TAG}"  when using a PIDX release
run "git fetch --depth=1 --prune origin ${PIDX_TAG}"
run "git checkout FETCH_HEAD"

run "cd build"

if test $2 != "no"; then
  DEBUG="-DCMAKE_BUILD_TYPE=Debug"
fi

run \
  "cmake \
    $DEBUG \
    -DCMAKE_INSTALL_PREFIX=\"$PIDX_DIR/install\" \
    -DPIDX_BUILD_PROFILE=FALSE \
    -DPIDX_BUILD_PROCESSING=FALSE \
    -DPIDX_BUILD_TUTORIAL=FALSE \
  .."

run "make -j4 install"

echo ""
echo "Done Building PIDX Library."
echo "------------------------------------------------------------------"
echo ""

# Return 0 == success
exit 0
