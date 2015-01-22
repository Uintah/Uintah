#! /bin/bash

host=`hostname`

echo
if test "$MACHINE" = ""; then
   echo "Please set the env var MACHINE to:"
   echo "  At Utah: Ember or Ash"
   echo "  At LLNL: Vulcan or Cab"
   echo "  At LANL: Mustang or Mapache"
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
   CC=icc
   CXX=icpc
   COMP=icc-15.0.0
else
   if test "$COMPILER" = "gcc"; then
      echo f  "Building with GCC"
      CC=gcc
      CXX=g++
      COMP=gcc-4.4.7
   else
      echo "ERROR: Env var COMPILER was set to '$COMPILER', but must be set to 'icc' or 'gcc'"
      echo
      exit
   fi
fi

TPI=Thirdparty-Install

echo
if test "$MACHINE" = "Ember"; then
  echo "  Building for Ember"
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

  echo "Building for Ash"
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
  BOOSTLOC=/uufs/$NAME/sys/pkg/boost/1.54.0_mvapich2-1.9
else
  echo ""
  echo "$MACHINE not supported yet... add it."
  echo ""
  exit
fi
fi

###########################################################################
#
# Some bullet proofing...
#

if test "$BOOSTLOC" = ""; then
   echo "ERROR: Env var BOOST_LOC \($BOOSTLOC\) not set or is not a valid directory..."
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

echo "  CC:   $CC"
echo "  CC:   $CXX"
echo "  COMP: $COMP"


INSTALL_BASE=$INSTALL_BASE-$DATE

echo 

