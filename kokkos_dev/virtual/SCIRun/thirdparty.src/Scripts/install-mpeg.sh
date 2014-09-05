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
    echo "ERROR: install-mpeg.sh: SHARED_LIBS value ($SHARED_LIBS) is invalid."
    echo ""
    exit
fi

if test "$BITS" = "64"; then 
    if test "$OSNAME" = "IRIX64"; then
        BITS_FLAG="-64"
    elif test "$OSNAME" = "AIX"; then
        BITS_FLAG="-q64"
    fi
else
    if test "$OSNAME" = "IRIX64"; then
        BITS_FLAG="-n32"
    elif test "$OSNAME" = "AIX"; then
        BITS_FLAG="-q32"
    fi
fi

echo
echo "Installing MPEG: $DIR $OSNAME $SH_LIB_EXT $BITS $TAR $SHARED_LIBS $MAKE_FLAGS"
echo

############
# Unpack:
	
install_source_dir=`pwd`
cd $DIR/src


if test -d $DIR/src/mpeg_encode; then
    if test "$DO_NOT_UNTAR" != "yes"; then
        echo "It appears that MPEG has already been untarred.  Do you wish to untar it again? (y/n)"
        read answer
        echo
        if test "$answer" = "y"; then
            $TAR zxf $install_source_dir/Tarballs/MPEGelib-0.3.tar.gz
            $TAR zxf $install_source_dir/Tarballs/mpeg_encode-1.5b-src.tar.gz
            mv -f MPEGelib-0.3/* mpeg_encode
            rm -rf MPEGelib-0.3
        fi
    fi
else
    $TAR zxf $install_source_dir/Tarballs/MPEGelib-0.3.tar.gz
    $TAR zxf $install_source_dir/Tarballs/mpeg_encode-1.5b-src.tar.gz
    mv -f MPEGelib-0.3/* mpeg_encode
    rm -rf MPEGelib-0.3
fi

############
# Patch:

# Note: -fPIC is patched in on Linux...

PATCH_FILE=$install_source_dir/Patches/$OSNAME'_'$BITS/mpeg_encode.patch

echo looking for $PATCH_FILE

if test -f "$PATCH_FILE"; then
    echo "#### PATCHING $OSNAME SPECIFIC MPEG FILES ####"
    echo
    sed 's,@INSTALL_DIR@,'$DIR',g' < $PATCH_FILE | patch -p0
    echo
    echo "#### DONE PATCHING FILES ####"
    echo
fi

############
# Make

cd $DIR/src/mpeg_encode

$MAKE $MAKE_FLAGS -f Makefile.lib
$MAKE -f Makefile.lib install
