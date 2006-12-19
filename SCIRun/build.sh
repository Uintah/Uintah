#!/bin/bash
#
# SCIRun 3.0.0 build script
#
# This script will build SCIRun from scratch
#
#
initialize() {
    if test "$1" = "Debug"; then
        buildtype="Debug"
        makeflags=$2
    else
        buildtype="Release"
        makeflags=$1
    fi

    if test `uname` = "Darwin"; then
        getcommand="curl -O"
        export LINK_X11=0
    elif test `uname` = "Linux"; then
        getcommand="wget"
        export LINK_X11=1
    else
        echo "Unsupported system.  Please run on OSX or Linux"
        exit 1
    fi
}

initialize $*

# will cause the script to bailout if the passed in command fails
try () {
  $*
  if [ $? != "0" ]; then
      echo -e "\n***ERROR in build script\nThe failed command was:\n$*\n"
      exit 1
  fi
}

# functionally equivalent to try(),
# but it prints a different error message
ensure () {
  $* >& /dev/null
  if [ $? != "0" ]; then
      echo -e "\n***ERROR, $* is required but not found on this system\n"
      exit 1
  fi
}

export DIR=`pwd`


# ensure make is on the system
ensure make --version

# Try to find a version of cmake
cmakebin=`which cmake`

#if it is not found
if [ ! -e "$cmakebin" ]; then
    # then look for our own copy made by this script previously
    cmakebin=$DIR/cmake/local/bin/cmake
    try mkdir -p $DIR/cmake/
    try cd $DIR/cmake
    if [ ! -e "$cmakebin" ]; then
        # try to downlaod and build our own copy in local
        try $getcommand http://www.cmake.org/files/v2.4/cmake-2.4.5.tar.gz
        try tar xvzf cmake-2.4.5.tar.gz
        try cd cmake-2.4.5
        try ./configure --prefix=$DIR/cmake/local
        try make $makeflags
        try make install
    fi
fi

echo cmakebin=$cmakebin
ensure $cmakebin --version

try cd $DIR/thirdparty.src
try rm -rf $DIR/thirdparty.bin/*
try mkdir -p $DIR/thirdparty.bin
try ./install.sh $DIR/thirdparty.bin $makeflags

echo `grep SCIRUN_THIRDPARTY_DIR $DIR/thirdparty.src/install_command.txt`
echo `grep SCIRUN_THIRDPARTY_DIR $DIR/thirdparty.src/install_command.txt | awk -F '\=' '{print $2}'`
export TP=`grep SCIRUN_THIRDPARTY_DIR $DIR/thirdparty.src/install_command.txt | awk -F '\=' '{print $2}'`
try echo "SCIRUN_THIRDPARTY_DIR=$TP"

if [ ! -d "$TP" ]; then
    echo "Thirdparty failed to build in $TP. Exiting"
    exit 1
fi

try cd $DIR/bin
try cmake ../src -DSCIRUN_THIRDPARTY_DIR=${TP}
try make $makeflags



