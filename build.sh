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
#    Author :
#    Date   :


# name? 0.1.1 build script
#
# This script will build name? from scratch
#

function usage() {
  echo -e "Usage: build.sh [--debug] [--no-gui] [--help] <path to SCIRun Thirdparty>\nSCIRun Thirdparty libraries available from www.sci.utah.edu."
  exit 0
}

## check and process args
if [ $# -lt 1 ] ; then
  usage
fi


## from SCIRun build script, author McKay Davis
if test `uname` = "Darwin"; then
  getcommand="curl -O"
  platform="darwin"
elif test `uname` = "Linux"; then
  getcommand="wget"
  platform="linux"
else
  echo "Unsupported system.  Please run on OSX or Linux"
  exit 1
fi

for a in $* ; do
  if [ $a = "--help" ] ; then
    usage
  elif [ $a = "--debug" ] ; then
    export DEBUG_BUILD=1
  elif [ $a = "--no-gui" ] ; then
    export NO_GUI=1
  else
    # check for SCIRun thirdparty
    if [ -d $a ] ; then
      export THIRDPARTY_INSTALL_DIR=$a
    else
      usage
    fi
  fi
done

export ROOT_DIR=`pwd`
export TEMP_DIR="build"

## from SCIRun build script, author McKay Davis
# will cause the script to bailout if the passed in command fails
function try() {
  $*
  if [ $? != "0" ]; then
      echo -e "\n***ERROR in build script\nThe failed command was:\n$*\n"
      exit 1
  fi
}

## from SCIRun build script, author McKay Davis
# functionally equivalent to try(),
# but it prints a different error message
function ensure() {
  $* >& /dev/null
  if [ $? != "0" ]; then
      echo -e "\n***ERROR, $* is required but not found on this system\n"
      exit 1
  fi
}


function getbabel() {
  build_dir="$ROOT_DIR/babel/local"
  try "$getcommand http://www.llnl.gov/CASC/components/docs/babel-1.0.2.tar.gz"
  try "tar xzvf babel-1.0.0.tar.gz"
  try "mkdir -p $build_dir"
  try "cd babel-1.0.0"
  if [ $platform = "darwin" ] ; then
    try "./configure --prefix=$ROOT_DIR/babel/local --disable-fortran77"
  else
    try "./configure --prefix=$ROOT_DIR/babel/local"
  fi
  try "make"
  try "make install"
}


# ensure make is on the system
ensure make --version
babelbin=`which babel`
wxconfigbin=`which wx-config`

export CONFIG_MIN="--enable-scirun2 --with-thirdparty=???"
