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
export GIT_SSL_NO_VERIFY=true
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
  # Building with CUDA
  THREADS="-DENABLE_THREADS=ON -DNTHREADS=1"
  if [[ `hostname` = titan* ]]; then
    CUDA="-DENABLE_CUDA=ON -DDISABLE_INTROSPECTION=ON -DCUDA_ARCHITECTURE_MINIMUM=\"3.5\" -DCUDA_HOST_COMPILER=/opt/gcc/4.9.0/bin/g++"
  else
    CUDA="-DENABLE_CUDA=ON"
  fi
else
  # Building without CUDA
  THREADS="-DENABLE_THREADS=ON -DNTHREADS=1"
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
#run "rm -rf install/*"

############################################################################
# SpatialOps

run "cd src"
needsrecompile=true
if [ -d "SpatialOps" ]; then
    run "cd SpatialOps"
    run "git remote update"

    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse origin/master)
    
    if [ $LOCAL = $REMOTE ]; then
        echo "SpatialOps is current - not rebuilding"
        needsrecompile=false
    else
      echo "updating SpatialOps..."
      run "git pull"
    fi
    run "cd .."
else
  run "git clone --depth 1 https://software.crsim.utah.edu:8443/James_Research_Group/SpatialOps.git SpatialOps"
  run "mkdir $BASE_BUILD_DIR/Wasatch3P/src/SpatialOps/build"
fi
if test ! -z $SPATIAL_OPS_TAG ; then
    run "cd SpatialOps"
    run "git reset --hard $SPATIAL_OPS_TAG"
    run "cd .."
fi

run "cd $BASE_BUILD_DIR/Wasatch3P/src/SpatialOps/build"

if [ "$needsrecompile" = true ]; then
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
    $THREADS \
    -DENABLE_TESTS=OFF \
    -DENABLE_EXAMPLES=OFF \
    $BOOST_FLAGS \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_HERE} \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    .."
  
  run "make -j4 install"
  
  if test "$6" = "yes"; then
  
    # Building with CUDA - Hack the SpatialOps installed NeboOperators.h file to cast the 2nd arg of pow to a double.
    echo ""
    echo ""
    echo "WARNING: Hacking NeboOperators.h to cast 2nd parameter of std::pow() to be a double!"
    echo ""
    echo ""
    run "mv $INSTALL_HERE/include/spatialops/NeboOperators.h $INSTALL_HERE/include/spatialops/NeboOperators.h.orig"
    run "sed -r -e 's,(std::pow.*)operand2_,\1(double)operand2_,' $INSTALL_HERE/include/spatialops/NeboOperators.h.orig > NeboOperators.h.tmp"
    run "sed -r -e 's,(std::pow.*)exp_,\1(double)exp_,' NeboOperators.h.tmp > $INSTALL_HERE/include/spatialops/NeboOperators.h"
    run "rm NeboOperators.h.tmp"
  
  fi
fi

run "cd ../../.."  # back to Wasatch3P

############################################################################
# ExprLib

run "cd src"
needsrecompile=true
if [ -d "ExprLib" ]; then
    run "cd ExprLib"
    run "git remote update"

    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse origin/master)

    if [ $LOCAL = $REMOTE ]; then
        echo "ExprLib is current - not rebuilding"
        needsrecompile=false
    else
      echo "updating ExprLib..."
      run "git pull"
    fi
    run "cd .."
else
    run "git clone --depth 20 https://software.crsim.utah.edu:8443/James_Research_Group/ExprLib.git ExprLib"
    run "mkdir $BASE_BUILD_DIR/Wasatch3P/src/ExprLib/build"
fi
if test ! -z $EXPR_LIB_TAG ; then
    run "cd ExprLib"
    run "git reset --hard $EXPR_LIB_TAG"
    run "cd .."
fi
run "cd $BASE_BUILD_DIR/Wasatch3P/src/ExprLib/build"

if [ "$needsrecompile" = true ]; then
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
fi

run "cd ../../.."  # back to Wasatch3P

############################################################################
# TabProps

run "cd src"
needsrecompile=true
if [ -d "TabProps" ]; then
    run "cd TabProps"
    run "git remote update"

    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse origin/master)

    if [ $LOCAL = $REMOTE ]; then
        echo "TabProps is current - not rebuilding"
        needsrecompile=false
    else
      echo "updating TabProps..."
      run "git pull"
    fi
    run "cd .."
else
    run "git clone --depth 1 https://software.crsim.utah.edu:8443/James_Research_Group/TabProps.git TabProps"
    run "mkdir $BASE_BUILD_DIR/Wasatch3P/src/TabProps/build"
fi
if test ! -z $TAB_PROPS_TAG ; then
    run "cd TabProps"
    run "git reset --hard $TAB_PROPS_TAG"
    run "cd .."
fi
run "cd $BASE_BUILD_DIR/Wasatch3P/src/TabProps/build"

if [ "$needsrecompile" = true ]; then
  
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
fi

run "cd ../../.."  # back to Wasatch3P

############################################################################
# RadProps

run "cd src"
needsrecompile=true
if [ -d "RadProps" ]; then
    run "cd RadProps"
    run "git remote update"

    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse origin/master)

    if [ $LOCAL = $REMOTE ]; then
        echo "RadProps is current - not rebuilding"
        needsrecompile=false
    else
      echo "updating RadProps..."
      run "git pull"
    fi
    run "cd .."
else
   run "git clone --depth 1 https://software.crsim.utah.edu:8443/James_Research_Group/RadProps.git RadProps"
   run "mkdir $BASE_BUILD_DIR/Wasatch3P/src/RadProps/build"
fi
if test ! -z $RAD_PROPS_TAG ; then
    run "cd RadProps"
    run "git reset --hard $RAD_PROPS_TAG"
    run "cd .."
fi

run "cd $BASE_BUILD_DIR/Wasatch3P/src/RadProps/build"

if [ "$needsrecompile" = true ]; then
  
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
fi

run "cd ../../.."  # back to Wasatch3P

############################################################################

echo ""
echo "Done Building Wasatch Thirdparty Libraries."
echo "------------------------------------------------------------------"
echo ""
export GIT_SSL_NO_VERIFY=false
# Return 0 == success
exit 0
