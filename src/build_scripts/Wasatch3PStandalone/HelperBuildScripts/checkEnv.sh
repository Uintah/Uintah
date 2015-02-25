#! /bin/bash

host=`hostname`

echo
if test "$MACHINE" = ""; then
   echo "Please set the env var MACHINE to:"
   echo "  At Utah: Ember, Ash, or Baja"
   echo "  At LLNL: Vulcan or Cab"
   echo "  At LANL: Mustang, Mapache, or Wolf"
   echo
   exit
fi

if ! test "$DATE"; then
  echo "Error, please set env var DATE for install dir suffix!"
  echo "   Something like: setenv DATE Dec_12_2014"
  exit
fi

if test "$COMPILER" = ""; then
   echo "Please set the env var COMPILER to icc or gcc!"
   echo
   exit
fi

if test "$COMPILER" = "icc"; then
   echo "  Building with ICC"
   CC=`which icc`
   CXX=`which icpc`
   COMP=icc14.0.4
else
   if test "$COMPILER" = "gcc"; then
      echo "  Building with GCC"
      CC=`which gcc`
      CXX=`which g++`
      COMP=gcc-4.4.7
   else
      echo "ERROR: Env var COMPILER was set to '$COMPILER', but must be set to 'icc' or 'gcc'"
      echo
      exit
   fi
fi

TPI=Thirdparty-Install

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
  NAME="ash.peaks"
  NAME2="Ash"
  TPI=thirdparty-install
  INSTALL_BASE=/uufs/$NAME/sys/pkg/uintah/$TPI$PHOENIXEXT/$NAME2/Wasatch3P
  BOOST_LOC=/uufs/$NAME/sys/pkg/boost/1.54.0_mvapich2-1.9
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
  echo "   * module load cmake/3.0.0"
  echo
  sleep 1

  NAME2="Mapache"
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
  echo "   * module load cmake/3.0.0"
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
  COMP=gcc4.9.1
  NAME2="Baja"
  INSTALL_BASE=/home/dav/thirdparty-install/$NAME2/Wasatch3P
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

export INSTALL_BASE=$INSTALL_BASE/build-$DATE

echo 

