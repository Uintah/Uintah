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
#    Author : Ayla Khan
#    Date   : 2007-02-01


# SCIJump 0.1.1 build script
#
# This script will build SCIJump from scratch
#

function usage() {
  echo -e "Usage: build.sh [OPTION...] SCIRun-Thirdparty-path"
  echo -e "Help options"
  echo -e "  --help, -h                 Show this help message."
  echo -e "Build options"
  echo -e "  --debug, -d                Build SCIJump in debug mode."
  echo -e "  --no-gui                   Build SCIJump without a gui."
  echo -e "  --mpi[=DIR]                Build SCIJump with MPI (optional path)."
  echo -e "  --enable-fortran77[=DIR]   Babel configure option: enables Fortran77 bindings."
  echo -e "  --enable-fortran90[=DIR]   Babel configure option: enables Fortran90 bindings."
  echo -e "\nThis script will configure and build SCIJump based on the options provided\nto this script.\nSee (code.sci.utah.edu/SCIJump/index.php/Main_Page) for more configuration options.\n"
  echo -e "This script will attempt to detect Babel (www.llnl.gov/CASC/components/babel.html)\nlibraries on your system.\nIf not found, version 1.0.2 will be downloaded and built.\n"
  echo -e "This script will also attempt to detect wxWidgets (www.wxwidgets.org) 2.6.x\non your system if configuring with a GUI.\nIf not found and if configuring with a GUI,\nversion 2.6.3 will be downloaded and built.\n"
  echo -e "To build SCIJump with parallel component support, use the --mpi option.\nIf an MPI implementation is not installed\nin standard system directories, provide the path.\nLAM-MPI and MPICH are supported.\n"
  echo -e "SCIRun Thirdparty libraries (required) are available for download from\nwww.sci.utah.edu."
  exit $1
}

## try function from SCIRun build script, author McKay Davis
# will cause the script to bailout if the passed in command fails
function try() {
  $*
  if [ $? != "0" ]; then
      echo -e "\n***ERROR in build script\nThe failed command was:\n$*\n"
      exit 1
  fi
}

## ensure function from SCIRun build script, author McKay Davis
# functionally equivalent to try(),
# but it prints a different error message
function ensure() {
  $* >& /dev/null
  if [ $? != "0" ]; then
      echo -e "\n***ERROR: $* is required but not found on this system\n"
      exit 1
  fi
}

## check and process args
if [ $# -lt 1 ] ; then
  usage 1
fi

## code snippet from SCIRun build script, author McKay Davis
if test `uname` = "Darwin"; then
  getcommand="curl -O"
  platform="darwin"
  sed_re="-E"
elif test `uname` = "Linux"; then
  getcommand="wget"
  platform="linux"
  sed_re="-r"
else
  echo "Unsupported system.  Please run on OSX or Linux"
  exit 1
fi

mpidir=
## Explicitly setting babel configure flags for fortran is a workaround
## for Babel issue 457.  A fix is expected in the Babel 1.0.4 release.
babel_fortan77_flags=
babel_fortan90_flags=
while [ "$1" != "" ] ; do
  case $1 in
    -h | --help )
      usage 0
      ;;
    --no-gui )
      export NO_GUI=1
      ;;
    -d | --debug )
      export DEBUG_BUILD=1
      ;;
    --mpi )
      export MPI_BUILD=1
      ;;
    --mpi=* )
      export MPI_BUILD=1
      mpidir=${1#--mpi=}
      ;;
    --enable-fortran77 )
      babel_fortran77_flags=$1
      ;;
    --enable-fortran90 )
      babel_fortran90_flags=$1
      ;;
    * )
      if [ -d $1 ] ; then
        export THIRDPARTY_INSTALL_DIR=$1
      else
        usage 1
      fi
      ;;
  esac
  shift
done

babel_fortran_flags=
if [ -z "$babel_fortan77_flags" ] ; then
  babel_fortran_flags="$babel_fortran_flags --disable-fortran77"
else
  babel_fortran_flags="$babel_fortran_flags $babel_fortran77_flags"
fi

if [ -z "$babel_fortan90_flags" ] ; then
  babel_fortran_flags="$babel_fortran_flags --disable-fortran90"
else
  babel_fortran_flags="$babel_fortran_flags $babel_fortran90_flags"
fi

if [ -z "$THIRDPARTY_INSTALL_DIR" ] ; then
  echo -e "***ERROR: missing path to SCIRun Thirdparty libraries.\n"
  usage 2
fi

export ROOT_DIR=`pwd`
export BUILD_DIR="build"

function getbabel() {
  build_dir="$ROOT_DIR/babel/local"
  babel_version="babel-1.0.2"
  babel_archive="$babel_version.tar.gz"

  if [ ! -e "$ROOT_DIR/$babel_version" ] ; then
    echo "***Downloading Babel 1.0.2***"
    try "$getcommand http://www.llnl.gov/CASC/components/docs/$babel_archive"
    try "tar xzvf $babel_archive"
  fi
  try "mkdir -p $build_dir"
  try "cd $babel_version"
  try "./configure --prefix=$build_dir $babel_fortran_flags"
  try "make"
  try "make install"
  try "cd $ROOT_DIR"
  export PATH="$PATH:$build_dir/bin"
  export BUILDDIR_TMP=$build_dir
}

function getwxwidgets_darwin() {
  wxwidgets_version="wxMac-2.6.3"
  wxwidgets_archive="$wxwidgets_version.tar.gz"
  build_dir="$ROOT_DIR/wxwidgets/local"

  if [ ! -e "$ROOT_DIR/wxwidgets_version" ] ; then
    echo "***Downloading wxMac 2.6.3***"
    try "$getcommand http://umn.dl.sourceforge.net/sourceforge/wxwindows/$wxwidgets_archive"
    try "tar xzvf $wxwidgets_archive"
  fi
  try "mkdir -p $build_dir"
  try "cd $wxwidgets_version"
  try "./configure --prefix=$build_dir --enable-shared --with-opengl --enable-threads"
  try "make"
  try "make install"
  try "cd $ROOT_DIR"
  export PATH="$PATH:$build_dir/bin"
  export BUILDDIR_TMP=$build_dir
}

function getwxwidgets_linux() {
  wxwidgets_version="wxGTK-2.6.3"
  wxwidgets_archive="$wxwidgets_version.tar.gz"
  build_dir="$ROOT_DIR/wxwidgets/local"

  if [ ! -e "$ROOT_DIR/wxwidgets_version" ] ; then
    echo "***Downloading wxGTK 2.6.3***"
    try "$getcommand http://umn.dl.sourceforge.net/sourceforge/wxwindows/$wxwidgets_archive"
    try "tar xzvf $wxwidgets_archive"
  fi
  try "mkdir -p $build_dir"
  try "cd $wxwidgets_version"
  try "./configure --prefix=$build_dir --enable-shared --with-opengl --enable-threads"
  try "make"
  try "make install"
  try "cd $ROOT_DIR"
  export PATH="$PATH:$build_dir/bin"
  export BUILDDIR_TMP=$build_dir
}

function versioncheck() {
  if [ ! -e "$1" -o -z "$2" ] ; then
    return 1
  else
    v=`$1 --version`
    ## cleanup possible junk in error messages...
    ## babel will give a warning if there are configure problems
    vc=`echo $v | tr -d [:cntrl:] | tr [:upper:] [:lower:] | grep 'warning'`
    if [ -n "$vc" ] ; then
      echo -e "Encountered problem after executing: $1 --version."
      return 1
    fi
    version=`echo $v | sed $sed_re 's/[A-Za-z ]//g'`
    ## split digits and compare
    major=`echo $version | sed 's/\.[0-9]*\.*[0-9]*//g'`
    if [ $major -lt $2 ] ; then
      return 1
    fi
    if [ -n "$3" ] ; then
      minor=`echo $version | sed 's/^[0-9]*\.//g' | sed $sed_re 's/\.[0-9]*\.*[0-9]*//g'`
      if [ $minor -lt $3 ] ; then
        return 1
      fi
    fi
    return 0
  fi
}

# ensure make is on the system
ensure make --version

# without gui
export CONFIG_ARGS="--enable-scijump --with-thirdparty=$THIRDPARTY_INSTALL_DIR"

if [ $DEBUG_BUILD ] ; then
  export CONFIG_ARGS="$CONFIG_ARGS --enable-debug"
fi

babelbin=`which babel`
babeldir=
versioncheck $babelbin 1 0
if [ $? == "0" ] ; then
  babelbindir=`dirname $babelbin`
  babeldir=`dirname $babelbindir`
else
  getbabel
  babeldir=$BUILDDIR_TMP
  unset BUILDDIR_TMP
fi

echo -e "Using Babel installation in $babeldir."
export CONFIG_ARGS="$CONFIG_ARGS --with-babel=$babeldir"

if [ ! $NO_GUI ] ; then
  wxconfigbin=`which wx-config`
  wxdir=
  versioncheck $wxconfigbin 2 6
  if [ $? == "0" ] ; then
    wxbindir=`dirname $wxconfigbin`
    wxdir=`dirname $wxbindir`
  else
    if test $platform = "darwin" ; then
      getwxwidgets_darwin
    else
      getwxwidgets_linux
    fi
    wxdir=$BUILDDIR_TMP
    unset BUILDDIR_TMP
  fi
  echo -e "Using wxWidgets installation in $wxdir."
  export CONFIG_ARGS="$CONFIG_ARGS --with-wxwidgets=$wxdir"
fi

if [ $MPI_BUILD ] ; then
  if [ -n "$mpidir" ] ; then
    export CONFIG_ARGS="$CONFIG_ARGS --with-mpi=$mpidir"
  else
    export CONFIG_ARGS="$CONFIG_ARGS --with-mpi"
  fi
fi

try "mkdir -p $BUILD_DIR"
try "cd $BUILD_DIR"
echo "***Configuring SCIJump***"
try "$ROOT_DIR/src/configure $CONFIG_ARGS"
try "make"
