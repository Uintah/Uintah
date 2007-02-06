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
      echo "Error: bad path to SCIRun Thirdparty libraries: $a"
      usage
    fi
  fi
done

if [ -z "$THIRDPARTY_INSTALL_DIR" ] ; then
  echo "Error: missing path to SCIRun Thirdparty libraries."
  usage
fi

export ROOT_DIR=`pwd`
export BUILD_DIR="build"

function getbabel() {
  build_dir="$ROOT_DIR/babel/local"
  babel_version="babel-1.0.2"
  babel_archive="$babel_version.tar.gz"
  try "$getcommand http://www.llnl.gov/CASC/components/docs/$babel_archive"
  try "tar xzvf $babel_archive"
  try "mkdir -p $build_dir"
  try "cd $babel_version"
  if [ $platform = "darwin" ] ; then
    try "./configure --prefix=$build_dir --disable-fortran77"
  else
    try "./configure --prefix=$build_dir"
  fi
  try "make"
  try "make install"
  try "cd $ROOT_DIR"
  return $build_dir
}

function getwxwidgets_darwin() {
  wxwidgets_version="wxMac-2.6.3"
  wxwidgets_archive="$wxwidgets_version.tar.gz"
  build_dir="$ROOT_DIR/wxwidgets/local"
  try "$getcommand http://prdownloads.sourceforge.net/wxwindows/$wxwidgets_archive"
  try "tar xzvf $wxwidgets_archive"
  try "mkdir -p $build_dir"
  try "cd $wxwidgets_version"
  try "./configure --prefix=$build_dir --enable-shared --enable-stl --enable-debug_gdb --enable-debug_flag --enable-debug_info --enable-debug_cntxt --enable-mem_tracing --enable-profile --with-opengl --enable-debug --enable-tabdialog --enable-std_string --enable-std_iostreams"
  try "make"
  try "make install"
  try "cd $ROOT_DIR"
  eval "$1=$build_dir"
}

function getwxwidgets_linux() {
  wxwidgets_version="wxGTK-2.6.3"
  wxwidgets_archive="$wxwidgets_version.tar.gz"

  build_dir="$ROOT_DIR/wxwidgets/local"
  try "$getcommand http://prdownloads.sourceforge.net/wxwindows/$wxwidgets_archive"
  try "tar xzvf $wxwidgets_archive"
  try "mkdir -p $build_dir"
  try "cd $wxwidgets_version"
  try "./configure --prefix=$build_dir --enable-shared --enable-stl --enable-debug_gdb --enable-debug_flag --enable-debug_info --enable-debug_cntxt --enable-mem_tracing --enable-profile --with-opengl --enable-debug --enable-tabdialog --enable-std_string --enable-std_iostreams"
  try "make"
  try "make install"
  try "cd $ROOT_DIR"
  eval "$1=$build_dir"
}

# ensure make is on the system
ensure make --version

# without gui
export CONFIG_ARGS="--enable-scirun2 --with-thirdparty=$THIRDPARTY_INSTALL_DIR"

if [ $DEBUG_BUILD ] ; then
  export CONFIG_ARGS="$CONFIG_ARGS --enable-debug"
fi

babelbin=`which babel`
babeldir=
if [ -e "$babelbin" ] ; then
  babelbindir=`dirname $babelbin`
  babeldir=`dirname $babelbindir`
else
  getbabel $babeldir
fi
echo -e "Using Babel installation in $babeldir."
export CONFIG_ARGS="$CONFIG_ARGS --with-babel=$babeldir"

if [ ! $NO_GUI ] ; then
  wxconfigbin=`which wx-config`
  wxdir=
  if [ -e "$wxconfigbin" ] ; then
    wxbindir=`dirname $wxconfigbin`
    wxdir=`dirname $wxbindir`
  else
    if test $platform = "darwin" ; then
      getwxwidgets_darwin $wxdir
    else
      getwxwidgets_linux $wxdir
    fi
  fi
  echo -e "Using wxWidgets installation in $wxdir."
  export CONFIG_ARGS="$CONFIG_ARGS --with-wxwidgets=$wxdir"
fi

try "cd $BUILD_DIR"
try "$ROOT_DIR/src/configure $CONFIG_ARGS"
try "make"
