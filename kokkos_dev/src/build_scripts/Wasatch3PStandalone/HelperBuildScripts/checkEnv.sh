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
   echo "  At Utah: Ember, Ash, or Baja"
   echo "  At LLNL: Vulcan, Cab, Surface, or Syrah"
   echo "  At LANL: Mustang, Mapache, or Wolf"
   echo "  At ORNL: titan"
   echo ""
   exit
fi

if ! test "$DATE"; then
  echo "Error, please set env var DATE for install dir suffix!"
  echo "   Something like: setenv DATE Dec_12_2015"
  exit
fi

if test "$COMPILER" = ""; then
   echo "Please set the env var COMPILER to icc, gcc, or xlc!"
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
      if test "$COMPILER" = "xlc"; then
         echo "  Building with xlc"
         CC=`which xlc`
         CXX=`which xlC`
         COMP=xlc-12.1
      else
         echo "ERROR: Env var COMPILER was set to '$COMPILER', but must be set to 'icc', 'gcc', or 'xlc'"
         echo
         exit
      fi
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
  #COMP=icc-16.0.0
  COMP=gcc-4.9.2
  NAME="ash.peaks"
  NAME2="Ash"
  TPI=thirdparty-install
  INSTALL_BASE=/uufs/$NAME/sys/pkg/uintah/$TPI$PHOENIXEXT/$NAME2/Wasatch3P
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
  COMP=gcc4.9.2
  NAME2="Baja"
  INSTALL_BASE=/home/dav/thirdparty-install/$NAME2/Wasatch3P
  BOOST_LOC=/usr
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
  CC=`which mpigcc`
  CXX=`which mpig++`
  COMP=gcc-4.4.7

  NAME2="Syrah"
  INSTALL_BASE=/usr/gapps/uintah/Thirdparty-install/syrah/Wasatch3P
  BOOST_LOC=/usr/gapps/uintah/Thirdparty-install/syrah/Boost/v1_55_0/mpigcc4.7.7-mvapich2.gnu.1.7
else
if test "$MACHINE" = "titan"; then
  
  if [[ $host != titan* ]]; then
     echo "Error: hostname did not return titan*... Goodbye."
     exit
  fi
  CC=/opt/cray/craype/2.4.0/bin/cc
  CXX=/opt/cray/craype/2.4.0/bin/CC
  COMP=cc-4.8.2
  NAME2="titan"
  INSTALL_BASE=/ccs/proj/csc188/utah/thirdparty-install/titan/Wasatch3P
  BOOST_LOC=$BOOST_ROOT
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

