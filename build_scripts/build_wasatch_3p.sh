#! /bin/bash

#
# This script is used by configure to build the thirdparty libraries required
# by the Wasatch component: SpatialOps, TabProps, ExprLib.
#
# $1 - location to build the 3P
# $2 - boost library location
# $3 - boost include location
# $4 - whether Uintah is being built in debug mode or not...
# $5 - whether Uintah is being built with static libraries or not ...
# $6 - whether Uintah is being built with CUDA
#

BASE_BUILD_DIR=$1
BOOST_LIBRARY=$2
BOOST_INCLUDE=$3

if test $4 != "no"; then
  DEBUG="-DCMAKE_BUILD_TYPE=Debug"
fi

if test $5 = "yes"; then
  STATIC="-DBoost_USE_STATIC_LIBS=ON"
else
  STATIC=""
fi

if test "$6" = "yes"; then
  if [[ `hostname` = titan* ]]; then
    CUDA="-DENABLE_CUDA=ON -DDISABLE_INTROSPECTION=ON -DCUDA_ARCHITECTURE_MINIMUM=\"3.5\" "
  else
    CUDA="-DENABLE_CUDA=ON"
  fi
else
  CUDA=""
fi



###########################################################################
# GIT Hash Tags for the various libraries

# SPATIAL_OPS_TAG=
# EXPR_LIB_TAG=
# TAB_PROPS_TAG=
# RAD_PROPS_TAG=

############################################################################

show_error()
{
    echo ""
    echo "An error occurred in the buid_wasatch script:"
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
echo   "Building Wasatch Thirdparty Libraries..."
echo   ""
if test -z "$BOOST_INCLUDE"; then
  echo "  Using Boost: Built In"
else
  echo "  Using Boost: Include: $BOOST_INCLUDE"
  echo "               Lib:     $BOOST_LIBRARY"
  BOOST_FLAGS="-DBOOST_INCLUDEDIR=$BOOST_INCLUDE -DBOOST_LIBRARYDIR=$BOOST_LIBRARY"
fi
echo   ""
echo   "  Using Cmake: "`which cmake`
echo   ""
echo   "------------------------------------------------------------------"
############################################################################
# Go to build/install directory

run "cd $BASE_BUILD_DIR/Wasatch3P"
run "mkdir -p src"
run "mkdir -p install"
run "rm -rf install/*"

############################################################################
# SpatialOps

run "cd src"
run "rm -rf SpatialOps"
run "env GIT_SSL_NO_VERIFY=true git clone --depth 1 https://software.crsim.utah.edu:8443/James_Research_Group/SpatialOps.git SpatialOps"
if test ! -z $SPATIAL_OPS_TAG ; then
    run "cd SpatialOps"
    run "git reset --hard $SPATIAL_OPS_TAG"
    run "cd .."
fi
run "mkdir $BASE_BUILD_DIR/Wasatch3P/src/SpatialOps/build"
run "cd $BASE_BUILD_DIR/Wasatch3P/src/SpatialOps/build"

# begin debugging
run "pwd"
run "ls -l"
run "which cmake"
run "cmake --version"
# end debugging

INSTALL_HERE=$BASE_BUILD_DIR/Wasatch3P/install/SpatialOps

run \
"cmake \
  $DEBUG \
  $STATIC \
  $CUDA \
  -DENABLE_TESTS=OFF \
  -DENABLE_THREADS=ON \
  -DNTHREADS=1 \
  -DENABLE_EXAMPLES=OFF \
  $BOOST_FLAGS \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_HERE} \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  .."

run "make -j4 install"
run "cd ../../.."  # back to Wasatch3P

############################################################################
# ExprLib

run "cd src"
run "rm -rf ExprLib"
run "env GIT_SSL_NO_VERIFY=true git clone --depth 20 https://software.crsim.utah.edu:8443/James_Research_Group/ExprLib.git ExprLib"
if test ! -z $EXPR_LIB_TAG ; then
    run "cd ExprLib"
    run "git reset --hard $EXPR_LIB_TAG"
    run "cd .."
fi
run "mkdir $BASE_BUILD_DIR/Wasatch3P/src/ExprLib/build"
run "cd $BASE_BUILD_DIR/Wasatch3P/src/ExprLib/build"

INSTALL_HERE=$BASE_BUILD_DIR/Wasatch3P/install/ExprLib
SPATIAL_OPS_INSTALL_DIR=$BASE_BUILD_DIR/Wasatch3P/install/SpatialOps

run                  \
"cmake               \
  $DEBUG             \
  $STATIC            \
  \
  -DENABLE_TESTS=OFF \
  -DBUILD_GUI=OFF    \
  \
  -DSpatialOps_DIR=${SPATIAL_OPS_INSTALL_DIR}/share \
  \
  $BOOST_FLAGS \
  \
  -DENABLE_UINTAH=ON \
  \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_HERE}            \
  -DCMAKE_CXX_FLAGS="-fPIC"                         \
  .."

run "make -j4 install"
run "cd ../../.."  # back to Wasatch3P

############################################################################
# TabProps

run "cd src"
run "rm -rf TabProps"
run "env GIT_SSL_NO_VERIFY=true git clone --depth 1 https://software.crsim.utah.edu:8443/James_Research_Group/TabProps.git TabProps"
if test ! -z $TAB_PROPS_TAG ; then
    run "cd TabProps"
    run "git reset --hard $TAB_PROPS_TAG"
    run "cd .."
fi
run "mkdir $BASE_BUILD_DIR/Wasatch3P/src/TabProps/build"
run "cd $BASE_BUILD_DIR/Wasatch3P/src/TabProps/build"

INSTALL_HERE=$BASE_BUILD_DIR/Wasatch3P/install/TabProps

run \
"cmake \
  $DEBUG \
  $STATIC \
  -DTabProps_PREPROCESSOR=OFF \
  -DTabProps_UTILS=OFF \
  -DTabProps_ENABLE_TESTING=OFF \
  $BOOST_FLAGS \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_HERE} \
  -DCMAKE_CXX_FLAGS=-fPIC \
  .."

run "make -j4 install"
run "cd ../../.."  # back to Wasatch3P

############################################################################
# RadProps

run "cd src"
run "rm -rf RadProps"
run "env GIT_SSL_NO_VERIFY=true git clone --depth 1 https://software.crsim.utah.edu:8443/James_Research_Group/RadProps.git RadProps"
if test ! -z $RAD_PROPS_TAG ; then
    run "cd RadProps"
    run "git reset --hard $RAD_PROPS_TAG"
    run "cd .."
fi
run "mkdir $BASE_BUILD_DIR/Wasatch3P/src/RadProps/build"
run "cd $BASE_BUILD_DIR/Wasatch3P/src/RadProps/build"

INSTALL_HERE=$BASE_BUILD_DIR/Wasatch3P/install/RadProps

run \
"cmake \
  $DEBUG \
  $STATIC \
  -DRadProps_ENABLE_TESTING=OFF \
  -DRadProps_ENABLE_PREPROCESSOR=OFF \
  $BOOST_FLAGS \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_HERE} \
  -DCMAKE_CXX_FLAGS=-fPIC \
  -DTabProps_DIR=${BASE_BUILD_DIR}/Wasatch3P/install/TabProps/share \
  .."

run "make -j4 install"
run "cd ../../.."  # back to Wasatch3P

############################################################################

echo ""
echo "Done Building Wasatch Thirdparty Libraries."
echo "------------------------------------------------------------------"
echo ""

# Return 0 == success
exit 0
