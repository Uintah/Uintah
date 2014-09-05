#!/bin/bash
#  
#  For more information, please see: http://software.sci.utah.edu
#  
#  The MIT License
#  
#  Copyright (c) 2006 Scientific Computing and Imaging Institute,
#  University of Utah.
#  
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#  
#    File   : build.sh
#    Author : McKay Davis
#    Date   : Tue Dec 19 15:35:50 2006


# SCIRun 3.0.0 build script
#
# This script will build SCIRun from scratch
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
ctestbin=`which ctest`

#if it is not found
if [ ! -e "$cmakebin" ]; then
    # then look for our own copy made by this script previously
    cmakebin=$DIR/cmake/local/bin/cmake
    ctestbin=$DIR/cmake/local/bin/ctest
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
echo ctestbin=$ctestbin
ensure $cmakebin --version

try cd $DIR/thirdparty.src
try rm -rf $DIR/thirdparty.bin/*
try mkdir -p $DIR/thirdparty.bin
try ./install.sh $DIR/thirdparty.bin $makeflags

try cd $DIR/bin

$cmakebin ../src

if [ -e "$ctestbin" ]; then
    try $ctestbin -VV -D Experimental -A $DIR/bin/CMakeCache.txt
else 
    try make $makeflags
fi


