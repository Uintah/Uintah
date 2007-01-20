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
    echo "ERROR: install-freetype.sh: SHARED_LIBS value ($SHARED_LIBS) is invalid."
    echo ""
    exit
fi

if test "$BITS" = "64"; then 
    if test "$OSNAME" = "IRIX64"; then
        CFLAGS="'-LANG:std -64'"
        CXXFLAGS=$CFLAGS
        LDFLAGS=$CFLAGS
    elif test "$OSNAME" = "AIX"; then
        CFLAGS="-q64"
        CXXFLAGS=$CFLAGS
        LDFLAGS=$CFLAGS
    else
        CFLAGS="-fPIC"
        CXXFLAGS=$CFLAGS
    fi
else
    if test "$OSNAME" = "IRIX64"; then
        CFLAGS="'-LANG:std -n32'"
        CXXFLAGS=$CFLAGS
        LDFLAGS=$CFLAGS
    elif test "$OSNAME" = "AIX"; then
        CFLAGS="-q32"
        CXXFLAGS=$CFLAGS
        LDFLAGS=$CFLAGS
    else
        CFLAGS="-fPIC"
        CXXFLAGS=$CFLAGS
    fi
fi

echo
echo "Installing freetype: $DIR $OSNAME $SH_LIB_EXT $BITS $TAR $SHARED_LIBS $MAKE_FLAGS"
echo

############
# Unpack:
	
install_source_dir=`pwd`
cd $DIR/src

if test -d $DIR/src/freetype-2.1.10; then
    if test "$DO_NOT_UNTAR" != "yes"; then
        echo "It appears that Freetype has already been untarred.  Do you wish to untar it again? (y/n)"
        read answer
        echo
        if test "$answer" = "y"; then
            $TAR zxf $install_source_dir/Tarballs/freetype-2.1.10.tar.gz
        fi
    fi
else
    $TAR zxf $install_source_dir/Tarballs/freetype-2.1.10.tar.gz
fi

############
# Patch:

# OS Patch
PATCH_FILE=$install_source_dir/Patches/$OSNAME'_'$BITS/freetype.patch

if test -f $PATCH_FILE; then
    echo "#### PATCHING $OSNAME SPECIFIC FREETYPE FILES ####"
    echo
    cat $PATCH_FILE | patch -p0
    echo
fi
   
############
# Configure: 

cd $DIR/src/freetype-2.1.10

# Specifying the GNUMAKE variable is necessary on SGI.
GNUMAKE=$MAKE ./configure $SHARED_LIBS_FLAG --prefix=$DIR $BITS_FLAG \
    CXXFLAGS="$CXXFLAGS" CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS"

############
# Make

$MAKE $MAKE_FLAGS
$MAKE install

