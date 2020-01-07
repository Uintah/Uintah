#! /bin/bash

host=`hostname`

echo

which cmake 2> /dev/null
if test "$?" != 0; then
   echo "Cmake not found..."
   echo
   exit
fi

echo
if test "$MACHINE" = ""; then
   echo "Please set the env var MACHINE to:"
   echo ""
   echo "  At Utah: Albion, Anasazi, Ash, Aurora, Baja, Cyrus, or Ember"
   echo "  At LLNL: Vulcan, Cab, Surface, or Syrah"
   echo "  At LANL: Mustang, Mapache, or Wolf"
   echo "  At ORNL: Titan"
   echo "  At Argonne: Mira"
   echo ""
   exit
fi

if ! test "$DATE"; then
  echo "ERROR: Please set env var DATE for install dir suffix!"
  echo "   Something like: setenv DATE Dec_12_2015"
  exit
fi

if test "$COMPILER" = ""; then
   echo "Make sure that the compiler specified in checkEvn.sh for your machine is set correctly - then setenv COMPILER to 'checked'."
   echo
   exit
fi

TPI=Thirdparty-Install

if test "$BUILD_CUDA" = ""; then
  echo "ERROR: Please set BUILD_CUDA to 'yes' or 'no'."
  echo
  exit
fi

if test "$BUILD_CUDA" = "yes"; then
  echo "Building with CUDA."
  CUDA_DIR_EXT="cuda"
else
  echo "NOT building with CUDA.  To turn CUDA on, set environment var BUILD_CUDA to 'yes'"
  CUDA_DIR_EXT="no_cuda"
fi
sleep 1

echo
echo "  Building for $MACHINE"

if test "$MACHINE" = "Ember"; then
  if [[ $host != ember* ]]; then
     echo "Error: hostname did not return ember... Goodbye."
     exit
  fi
  NAME="ember.arches"
  NAME2="Ember"
  TPI=thirdparty-install
  INSTALL_BASE=/uufs/$NAME/sys/pkg/uintah/$TPI/$NAME2/Wasatch3P
  BOOST_LOC=/uufs/$NAME/sys/pkg/boost/1.54.0_mvapich2-1.9
else
if test "$MACHINE" = "Ash"; then

  if test "$PHOENIX" = ""; then
    echo
    echo "ERROR: Please set the env var PHOENIX to yes or no!"
    echo
    exit
  fi

  if test ! "$PHOENIX" = "yes" -a ! "$PHOENIX" = "no" ; then
    echo 
    echo "ERROR: Please set the env var PHOENIX to 'yes' or 'no' - it is currently '$PHOENIX'!"
    echo
    exit
  fi

  if test "$PHOENIX" = "yes"; then
    echo "    Building for Phoenix"
    echo
    PHOENIXEXT="-phoenix"
  else
    PHOENIXEXT=""
  fi

  if [[ $host != ash* ]]; then
     echo
     echo "ERROR: hostname did not return ash... Goodbye."
     exit
  fi
  #COMP=icc-16.0.2
  COMP=gcc-4.9.2
  CC=`which mpicc`
  CXX=`which mpic++`
  NAME="ash.peaks"
  NAME2="Ash"
  TPI=thirdparty-install
  INSTALL_BASE=/uufs/$NAME/sys/pkg/uintah/$TPI$PHOENIXEXT/$NAME2/Wasatch3P
  #BOOST_LOC=/uufs/ash.peaks/sys/pkg/uintah/thirdparty-install/Boost/v1_60_0/icc16.0.2
  BOOST_LOC=/uufs/chpc.utah.edu/sys/installdir/boost/1.59.0-4.9.2g
else
if test "$MACHINE" = "Mapache"; then
  if [[ $host != mp-fe* ]]; then
     echo "Error: hostname did not return mp-fe*... Goodbye."
     exit
  fi
  #COMP=gcc4.7.2
  echo
  echo "Have you run the appropriate 'module load' commands? Eg:"
  echo "   * module load gcc"
  echo "   * module load cmake/3.1.0"
  echo
  sleep 1

  NAME2="Mapache"
  INSTALL_BASE=/usr/projects/uintah/Thirdparty-Install/$NAME2/Wasatch3P
  BOOST_LOC=/usr/projects/uintah/Thirdparty-Install/$NAME2/Boost/v1_56_0-$COMP
else
if test "$MACHINE" = "Mustang"; then
  if [[ $host != mu-fe* ]]; then
     echo "Error: hostname did not return mu-fe*... Goodbye."
     exit
  fi
  COMP=gcc4.7.2
  echo
  echo "Have you run the appropriate 'module load' commands? Eg:"
  echo "   * module load gcc"
  echo "   * module load cmake/3.1.0"
  echo
  sleep 1

  NAME2="Mustang"
  INSTALL_BASE=/usr/projects/uintah/Thirdparty-Install/$NAME2/Wasatch3P
  BOOST_LOC=/usr/projects/uintah/Thirdparty-Install/$NAME2/Boost/v1_56_0-$COMP
else
if test "$MACHINE" = "Wolf"; then
  if [[ $host != wf-fe* ]]; then
     echo "Error: hostname did not return wf-fe*... Goodbye."
     exit
  fi
  #COMP=gcc4.7.2
  echo
  echo "Have you run the appropriate 'module load' commands? Eg:"
  echo "   * module load intel/14.0.4"
  echo "   * module load cmake/3.1.0"
  echo
  sleep 1

  NAME2="Wolf"
  INSTALL_BASE=/usr/projects/uintah/Thirdparty-Install/$NAME2/Wasatch3P
  BOOST_LOC=/usr/projects/uintah/Thirdparty-Install/$NAME2/Boost/v1_56_0-$COMP
else
if test "$MACHINE" = "Baja"; then
  if [[ $host != baja* ]]; then
     echo "Error: hostname did not return baja*... Goodbye."
     exit
  fi
  CC=`which gcc`
  CXX=`which g++`
  COMP=gcc4.9.2
  NAME2="Baja"
  INSTALL_BASE=/home/dav/thirdparty-install/$NAME2/Wasatch3P
  BOOST_LOC=/usr
else
if test "$MACHINE" = "Aurora"; then
  if [[ $host != aurora* ]]; then
     echo "Error: hostname did not return aurora*... Goodbye."
     exit
  fi
  CC=`which gcc`
  CXX=`which g++`
  COMP=gcc5.3.1
  NAME2="Aurora"
  INSTALL_BASE=/uufs/chpc.utah.edu/common/home/u0080076/Thirdparty-Install/$NAME2/Wasatch3P
  BOOST_LOC=/usr
else
if test "$MACHINE" = "Anasazi"; then
  if [[ $host != anasazi* ]]; then
     echo "Error: hostname did not return anasazi*... Goodbye."
     exit
  fi
  # These don't work, but in order to see the path to boost I had to use them:
  #
  #    module load gcc/4.9.2
  #    module load boost
  #
  CC=`which gcc`
  CXX=`which g++`
  COMP=gcc4.9.2
  NAME2="Anasazi"
  INSTALL_BASE=/uufs/chpc.utah.edu/common/home/u0080076/Thirdparty-Install/$NAME2/Wasatch3P
  BOOST_LOC=/uufs/chpc.utah.edu/sys/installdir/boost/1.59.0-4.9.2g
else
if test "$MACHINE" = "Vulcan"; then
  if [[ $host != vulcanlac* ]]; then
     echo "Error: hostname did not return vulcanlac*... Goodbye."
     exit
  fi
  CC=`which bggcc-4.7.2`
  CXX=`which bgg++-4.7.2`
  COMP=bggcc4.7.2

  NAME2="Vulcan"
  INSTALL_BASE=/usr/gapps/uintah/Thirdparty-install/vulcan/Wasatch3P
  BOOST_LOC=/usr/gapps/uintah/Thirdparty-install/vulcan/Boost/v1_55_0-$COMP
else
if test "$MACHINE" = "Surface"; then
  if [[ $host != surface* ]]; then
     echo "Error: hostname did not return surface*... Goodbye."
     exit
  fi
  CC=`which mpigcc`
  CXX=`which mpig++`
  COMP=gcc-4.9.2

  NAME2="Surface"
  INSTALL_BASE=/usr/gapps/uintah/Thirdparty-install/surface/Wasatch3P
  BOOST_LOC=/usr/gapps/uintah/Thirdparty-install/surface/Boost/v1_55_0
else
if test "$MACHINE" = "Syrah"; then
  if [[ $host != syrah* ]]; then
     echo "Error: hostname did not return syrah*... Goodbye."
     exit
  fi
  CC=`which mpicc`
  CXX=`which mpic++`
  COMP=icc-16.0.1

  NAME2="Syrah"
  INSTALL_BASE=/usr/gapps/uintah/Thirdparty-install/syrah/Wasatch3P
  BOOST_LOC=/usr/gapps/uintah/Thirdparty-install/syrah/Boost/v1_60_0/intel16
else
if test "$MACHINE" = "Titan"; then
  
  if [[ $host != titan* ]]; then
     echo "Error: hostname did not return titan*... Goodbye."
     exit
  fi
  CC=`which cc`
  CXX=`which CC`
  COMP=cc-4.9.3
  NAME2="titan"
  INSTALL_BASE=/ccs/proj/csc188/utah/thirdparty-install/titan/Wasatch3P
  BOOST_LOC=$BOOST_ROOT
  #BOOST_LOC=/ccs/proj/csc188/utah/thirdparty-install/titan/Boost/v1_57_0/cc4.9.0-mpich7.2.5
else
if test "$MACHINE" = "Mira"; then
  
  if [[ $host != mira* ]]; then
     echo "Error: hostname did not return mira*... Goodbye."
     exit
  fi
 CC=`which mpicc`
 CXX=`which mpic++`
# COMP=bgclang3.9
# BOOST_LOC=/soft/libraries/boost/1.61.0/cnk-bgclang++11/current

#  CC=/soft/compilers/wrappers/gcc/mpicc
#  CXX=/soft/compilers/wrappers/gcc/mpic++
#  COMP=xlc-12.1
#  BOOST_LOC=/soft/libraries/boost/1.55.0/cnk-xl/current

  BOOST_LOC=/soft/libraries/boost/1.61.0/cnk-gcc-4.7.2/current
  COMP=gcc-4.8.4

  NAME2="Mira"
  INSTALL_BASE=/gpfs/mira-fs1/projects/SoPE_2/utah/pkgs/gcc/Wasatch3P
else
if test "$MACHINE" = "Albion"; then
  
  if [[ $host != albion* ]]; then
     echo "Error: hostname did not return albion*... Goodbye."
     exit
  fi
  CC=/usr/bin/gcc
  CXX=/usr/bin/g++
  COMP=gcc-4.9.2
  NAME2="albion"
  INSTALL_BASE=/home/dav/thirdparty-install/$NAME2/Wasatch3P
  BOOST_LOC=/usr/local/boost
else
if test "$MACHINE" = "Cyrus"; then
  if [[ $host != cyrus* ]]; then
     echo "Error: hostname did not return cyrus*... Goodbye."
     exit
  fi
  CC=/usr/bin/gcc
  CXX=/usr/bin/g++
  COMP=gcc-4.8.4
  NAME2="cyrus"
  INSTALL_BASE=/raid/home/harman/thirdparty-install/$NAME2/Wasatch3P
  BOOST_LOC=/usr
else
  echo ""
  echo "$MACHINE not supported yet... add it."
  echo ""
  exit
fi
fi
fi
fi
fi
fi
fi
fi
fi
fi
fi
fi
fi
fi
fi

###########################################################################
#
# Some bullet proofing...
#

if test "$BOOST_LOC" = "" -o ! -d "$BOOST_LOC"; then
   echo "ERROR: BOOST_LOC ($BOOST_LOC) not set or is not a valid directory..."
   echo "Exiting..."
   echo
   exit
fi

if test ! -d "$INSTALL_BASE"; then
   echo "ERROR: INSTALL_BASE ($INSTALL_BASE) is not a valid directory..."
   echo "Exiting..."
   echo
   exit
fi

############################################################################

# Add date to complete installation dir variable:

export CC
export CXX
export COMP
export BOOST_LOC

echo "  CC:    $CC"
echo "  CXX:   $CXX"
echo "  COMP:  $COMP"
echo "  BOOST: $BOOST_LOC"

export INSTALL_BASE=$INSTALL_BASE/build-$DATE-$CUDA_DIR_EXT

echo 

