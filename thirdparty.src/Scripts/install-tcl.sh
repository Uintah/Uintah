#
# parms INSTALL_DIR
#       OSNAME       [eg: Darwin]
#       SH_LIB_EXT   [eg: dylib]
#       BITS         [eg: 32|64]
#       TAR          [eg: gtar, a Gnu tar program]
#       SHARED_LIBS  [eg: DEFAULT|ENABLE|DISABLE]
#       MAKE_FLAGS   [eg: -j n]
#       

DIR=$1
OSNAME=$2
SH_LIB_EXT=$3
BITS=$4
TAR=$5
SHARED_LIBS=$6
MAKE_FLAGS=$7

if test "$SHARED_LIBS" = "DEFAULT"; then 
    SHARED_LIBS_FLAG=
elif test "$SHARED_LIBS" = "ENABLE"; then 
    SHARED_LIBS_FLAG="--enable-shared"
elif test "$SHARED_LIBS" = "DISABLE"; then 
    SHARED_LIBS_FLAG="--disable-shared"
else
    echo ""
    echo "ERROR: install-tcl.sh: SHARED_LIBS value ($SHARED_LIBS) is invalid."
    echo ""
    exit
fi

if test "$BITS" = "64"; then 
    BITS_FLAG="--enable-64bit"
else
    BITS_FLAG=
fi

echo
echo "Installing TCL: $DIR $OSNAME $SH_LIB_EXT $BITS $TAR $SHARED_LIBS $MAKE_FLAGS"
echo

############
# Unpack:
	
install_source_dir=`pwd`
cd $DIR/src

if test -d $DIR/src/tcl8.3.4; then
    if test "$DO_NOT_UNTAR" != "yes"; then
        echo "It appears that TCL has already been untarred.  Do you wish to untar it again? (y/n)"
        read answer
        echo
        if test "$answer" = "y"; then
            $TAR zxf $install_source_dir/Tarballs/tcl8.3.4.tar.gz
        fi
    fi
else
    $TAR zxf $install_source_dir/Tarballs/tcl8.3.4.tar.gz
fi

if test ! -L tcl; then
   ln -fs tcl8.3.4 tcl
fi

############
# Patch:

echo
echo "#### PATCHING TCL FILES ####"
echo

if test -f $DIR/src/tcl/unix/tclUnixNotfy.c.orig; then
    # This seems to happen on the SGI as, I guess, their 'patch' program isn't that smart.
    echo "Looks like patch has already been applied... skipping."
    echo
else

    # Generic Patch
    PATCH_FILE=$install_source_dir/Patches/Generic/tcl8.3.4_notify.patch 

    sed 's,<PREFIX>, '$DIR' ,g' < $PATCH_FILE | patch -p0
    echo

    # OS Patch
    PATCH_FILE=$install_source_dir/Patches/$OSNAME'_'$BITS/tcl8.3.4.patch

    if test -f $PATCH_FILE; then
        echo "#### PATCHING $OSNAME SPECIFIC TCL FILES ####"
        echo
        cat $PATCH_FILE | patch -p0
        echo
    fi
fi
   
echo "#### DONE PATCHING FILES ####"
echo

############
# Configure: 

cd $DIR/src/tcl/unix

./configure $SHARED_LIBS_FLAG --prefix=$DIR $BITS_FLAG

############
# Update Makefiles

if test "$BITS" = "64" && test "$OSNAME" = "IRIX64"; then 
    echo
    echo "#### FIXING MAKEFILE FOR 64bit SGI BUILD ####"
    echo
    # TCL's configure, on SGI IRIX, can't figure out the '-64' flag,
    # so we add it manually to the Makefile.

    mv Makefile Makefile.old
    sed -e "s/-n32/-64/g" Makefile.old > Makefile

    echo
    echo "#### DONE                                ####"
    echo
fi

############
# Make

$MAKE $MAKE_FLAGS
$MAKE install

############
# Copy needed include files into main 'include' directory.

cd $DIR
cp src/tcl/generic/tclInt.h include
cp src/tcl/generic/tclIntDecls.h include

############
# Add necessary symbolic links

cd $DIR/lib
if test ! -L libtcl.$SH_LIB_EXT && test -e libtcl8.3.$SH_LIB_EXT; then
   ln -fs libtcl8.3.$SH_LIB_EXT libtcl.$SH_LIB_EXT
fi
if test ! -L tcl && test -e tcl8.3; then
   ln -fs tcl8.3 tcl
fi

