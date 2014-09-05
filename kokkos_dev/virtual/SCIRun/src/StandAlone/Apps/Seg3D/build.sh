#!/bin/bash
#
# This script will download and build all packages for Seg3D
#
# For more info, see http://seg3d.org
# 
# Copyright 2006 Scientific Computing and Imaging Institute, University of Utah
# http://www.sci.utah.edu
#
# Author: McKay Davis
# Date: Oct 19, 2006
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
        exit
    fi
}

initialize $*

# will cause the script to bailout if the passed in command fails
try () {
  $*
  if [ $? != "0" ]; then
      echo -e "\n***ERROR in build script\nThe failed command was:\n$*\n"
      exit
  fi
}

# functionally equivalent to try(),
# but it prints a different error message
ensure () {
  $* >& /dev/null
  if [ $? != "0" ]; then
      echo -e "\n***ERROR, $* is required but not found on this system\n"
      exit
  fi
}


try export DIR=`pwd`/seg3d-build
#if [ -d $DIR ]; then
#    echo "$DIR exists. remove it and re-run script"
#    exit
#fi
try mkdir -p $DIR
try cd $DIR

# ensure cvs is on the system
ensure cvs --version

# ensure cvs is on the system
ensure make --version

# Try to find a version of svn
svnbin=`which svn`

# If its not found, 
if [ ! -e "$svnbin" ]; then
    # Look for our own copy made by this script previously
    svnbin=$DIR/local/bin/svn
    if [ ! -e "$svnbin" ]; then
        # Try to download and build our own copy in local
        try $getcommand http://subversion.tigris.org/downloads/subversion-1.1.4.tar.gz
        try tar zxvf subversion-1.1.4.tar.gz
        try cd subversion-1.1.4
        try ./configure --with-ssl --prefix=$DIR/local
        try make
        try make install
    fi
fi

echo svnbin=$svnbin
ensure $svnbin --version

# Try to find a version of cmake
cmakebin=`which cmake`

#if it is not found
if [ ! -e "$cmakebin" ]; then
    # then look for our own copy made by this script previously
    cmakebin=$DIR/local/bin/cmake
    if [ ! -e "$cmakebin" ]; then
        # try to downlaod and build our own copy in local
        try $getcommand http://www.cmake.org/files/v2.4/cmake-2.4.3.tar.gz
        try tar xvzf cmake-2.4.3.tar.gz
        try cd cmake-2.4.3
        try ./configure --prefix=$DIR/local
        try make $makeflags
        try make install
    fi
fi

echo cmakebin=$cmakebin
ensure $cmakebin --version

# Get the latest SCIRun Core Thirdparty
try cd $DIR

if [ -d "$DIR/3P" ]; then
    try cd $DIR/3P
    tpurl=`$svnbin info . | grep URL | cut -d" " -f 2`
    local=`$svnbin info . | grep Revision | cut -d" " -f 2`
    remote=`$svnbin info $tpurl | grep Revision | cut -d" " -f 2`
    try echo "local=$local remote=$remote tpurl=$tpurl"
    if test "$local" = "$remote"; then
        rebuldtp=0
    else
        try cd $DIR/3P
        try $svnbin update
        rebuildtp=1
    fi
else
    tpurl="https://code.sci.utah.edu/svn/Thirdparty/3.0.0"
    try $svnbin co $tpurl 3P
    rebuildtp=1
fi

if test "$rebuildtp" = "1"; then
    # rebuild the Thirdparty from scratch
    try rm -rf $DIR/Thirdparty
    try mkdir -p $DIR/Thirdparty
    try cd $DIR/3P
    try ./install.sh --seg3d-only $DIR/Thirdparty $makeflags
fi

try export TP=`cat $DIR/3P/thirdparty_dir`
try echo "Thirdparty = $TP"
try cd $TP
try rm -rf $TP/lib/*.dylib $TP/lib/*.so


# Get the latest CVS repository version of Insight Toolkit
try cd $DIR
if [ -d "$DIR/Insight" ]; then
    cd $DIR/Insight
    try cvs update
else
    try cvs -d :pserver:anonymous:insight@www.itk.org:/cvsroot/Insight co Insight
fi

# Build Insight Toolkit in Insight-bin w/ cmake
try mkdir -p $DIR/Insight-bin
try cd $DIR/Insight-bin
try $cmakebin $DIR/Insight -DBUILD_EXAMPLES=0 -DBUILD_TESTING=0 -DCMAKE_BUILD_TYPE=$buildtype -DCMAKE_INSTALL_PREFIX=$TP
try make $makeflags
try make install
# ensure that the insight installation suceeded
try cd $TP/lib/InsightToolkit

# Get the latest SVN version of SCIRun Core
try cd $DIR
if [ -d "$DIR/src" ]; then
    try cd $DIR/src
    try $svnbin update
else
    try $svnbin co https://code.sci.utah.edu/svn/SCIRun/branches/cibc/src src
fi

# Build SCIRun Core w/ cmake in SCIRunCore-bin
try mkdir -p $DIR/bin
try cd $DIR/bin
try $cmakebin $DIR/src -DSCIRUN_THIRDPARTY_DIR=$TP -DCMAKE_BUILD_TYPE=$buildtype -DWITH_X11=$LINK_X11 -DBUILD_SHARED_LIBS=0 -DBUILD_DATAFLOW=0 -DBUILD_SEG3D=1
try make $makeflags
