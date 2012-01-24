#! /bin/sh

#
# This script is used by configure to build the thirdparty libraries required
# by the Wasatch component: SpatialOps, TabProps, ExprLib.
#
# $1 - location to build the 3P
# $2 - boost location
# $3 - whether Uintah is being built in debug mode or not...
#

BASE_BUILD_DIR=$1
BOOST_DIR=$2

if test $3 != "no"; then
  DEBUG="-DCMAKE_BUILD_TYPE=Debug"
fi

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

echo ""
echo "------------------------------------------------------------------"
echo "Building Wasatch Thirdparty Libraries..."
echo ""
echo "  Using Boost: $BOOST_DIR"
echo ""

############################################################################
# Go to build/install directory

run "cd $BASE_BUILD_DIR/Wasatch3P"
run "mkdir -p src"
run "mkdir -p install"

############################################################################
# SpatialOps

run "cd src"
run "rm -rf SpatialOps"
run "git clone git://software.crsim.utah.edu/SpatialOps"
run "cd $BASE_BUILD_DIR/Wasatch3P/src/SpatialOps"

INSTALL_HERE=$BASE_BUILD_DIR/Wasatch3P/install/SpatialOps

run \
"cmake \
  $DEBUG \
  -DENABLE_TESTS=OFF \
  -DBOOST_ROOT=$BOOST_DIR \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_HERE} \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  ."

run "make -j4"
run "make install"

############################################################################
# ExprLib

run "cd .."
run "rm -rf ExprLib"
run "git clone git://software.crsim.utah.edu/ExprLib"
run "cd $BASE_BUILD_DIR/Wasatch3P/src/ExprLib"

INSTALL_HERE=$BASE_BUILD_DIR/Wasatch3P/install/ExprLib
SPATIAL_OPS_INSTALL_DIR=$BASE_BUILD_DIR/Wasatch3P/install/SpatialOps

run \
"cmake \
  $DEBUG \
  -DENABLE_TESTS=OFF \
  -DBOOST_ROOT=$BOOST_DIR \
  \
  -DSpatialOps_DIR=${SPATIAL_OPS_INSTALL_DIR}/share \
  \
  -DENABLE_UINTAH=ON \
  -DENABLE_THREADS=OFF \
  \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_HERE} \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  ."

run "make -j4"
run "make install"

############################################################################
# TabProps

run "cd .."
run "rm -rf TabProps"
run "git clone git://software.crsim.utah.edu/TabProps"
run "cd $BASE_BUILD_DIR/Wasatch3P/src/TabProps"

INSTALL_HERE=$BASE_BUILD_DIR/Wasatch3P/install/TabProps

run \
"cmake \
  $DEBUG \
  -DTabProps_PREPROCESSOR=OFF \
  -DBoostOutput=ON \
  -DBOOST_ROOT=$BOOST_DIR \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_HERE} \
  -DCMAKE_CXX_FLAGS=-fPIC \
  -DCMAKE_CXX_LINK_FLAGS=\"-lpthread -lz\" \
  ."

run "make -j4"
run "make install"

############################################################################

echo ""
echo "Done Building Wasatch Thirdparty Libraries."
echo "------------------------------------------------------------------"
echo ""

# Return 0 == success
exit 0
