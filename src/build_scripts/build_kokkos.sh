#! /bin/bash

#
# This script is used by Uintah's configure to build the Kokkos thirdparty library
#
# $1 - location to build the 3P

KOKKOS_TAG="2.7.00"

KOKKOS_DIR=$1

############################################################################

show_error()
{
    echo ""
    echo "An error occurred in the build_kokkos.sh script:"
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
echo   "Building Kokkos Library..."
echo   ""
echo   ""
echo   "    Using Cmake: "`which cmake`
echo   ""
echo   "------------------------------------------------------------------"

if  [ ! -d "$KOKKOS_DIR" ]; then
  run "git clone https://github.com/kokkos/kokkos.git $KOKKOS_DIR/src"
fi

run "cd $KOKKOS_DIR/src"

run "git checkout $KOKKOS_TAG"

run "cd .."

if  [ ! -d "./build" ]; then
  run "mkdir build"
fi

run "cd build"

run \
  "$KOKKOS_DIR/src/generate_makefile.bash \
    --prefix=$KOKKOS_DIR/build \
    --kokkos-path=$KOKKOS_DIR/src \
    --with-openmp \
    --with-serial \
    --cxxflags="" "

run "make -j4 CC=gcc CXX=g++ CXXFLAGS="" LINK=g++ | tee out.kokkos.compile"

run "make install CC=gcc CXX=g++ CXXFLAGS="" LINK=g++"

echo ""
echo "Done Building Kokkos Library."
echo "------------------------------------------------------------------"
echo ""

# Return 0 == success
exit 0
