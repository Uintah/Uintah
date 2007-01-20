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
    echo "ERROR: install-tk.sh: SHARED_LIBS value ($SHARED_LIBS) is invalid."
    echo ""
    exit
fi

if test "$BITS" = "64"; then 
    BITS_FLAG="--enable-64bit"
else
    BITS_FLAG=
fi

echo
echo "Installing TK: $DIR $OSNAME $SH_LIB_EXT $BITS $TAR $SHARED_LIBS $MAKE_FLAGS"
echo

############
# Unpack:
	
install_source_dir=`pwd`
cd $DIR/src

if test -d $DIR/src/tk8.3.4; then
    if test "$DO_NOT_UNTAR" != "yes"; then
        echo "It appears that TK has already been untarred.  Do you wish to untar it again? (y/n)"
        read answer
        echo
        if test "$answer" = "y"; then
            $TAR zxf $install_source_dir/Tarballs/tk8.3.4.tar.gz
        fi
    fi
else
    $TAR zxf $install_source_dir/Tarballs/tk8.3.4.tar.gz
fi

if test ! -L tk; then
   ln -fs tk8.3.4 tk
fi

############
# Patch:

echo
echo "#### PATCHING TK FILES ####"
echo

if test -f $DIR/src/tk/unix/configure.orig; then
    # This seems to happen on the SGI as, I guess, their 'patch' program isn't that smart.
    echo "Looks like patch has already been applied... skipping."
    echo
else
    # Generic Patch
    echo "  #### Generic TK Patch ####"
    echo
    PATCH_FILE=$install_source_dir/Patches/Generic/tk8.3.4.patch
    cat $PATCH_FILE | patch -p0

    PATCH_FILE=$install_source_dir/Patches/$OSNAME'_'$BITS/tk8.3.4.patch
    if test -f $PATCH_FILE; then
        echo 
        echo "  #### $OSNAME Specific TK Patch ####"
        echo
        cat $PATCH_FILE | patch -p0
    fi
fi
   
echo
echo "#### DONE PATCHING FILES ####"
echo

############
# Configure: 

cd $DIR/src/tk/unix

./configure $SHARED_LIBS_FLAG --prefix=$DIR $BITS_FLAG

############
# Update Makefiles

if test "$BITS" = "64" && test "$OSNAME" = "IRIX64"; then 
    echo
    echo "#### FIXING MAKEFILE FOR 64bit SGI BUILD ####"
    echo

    # This assumes that the Makefile was built on an SGI... on
    # anything else it won't have a -n32 so this does nothing there.
    # Note, on 64 bit linux, you probably don't need a flag (it just
    # defaults to 64 bit mode), so you don't need to add a flag to the
    # Makefile...

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
# Add necessary symbolic links

cd $DIR/lib
if test ! -L libtk.$SH_LIB_EXT && test -e libtk8.3.$SH_LIB_EXT; then
   ln -fs libtk8.3.$SH_LIB_EXT libtk.$SH_LIB_EXT
fi
if test ! -L tk && test -e tk8.3; then
   ln -fs tk8.3 tk
fi


