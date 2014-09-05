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
    echo "ERROR: install-itcl.sh: SHARED_LIBS value ($SHARED_LIBS) is invalid."
    echo ""
    exit
fi

if test "$BITS" = "64"; then 
    BITS_FLAG="--enable-64bit"
else
    BITS_FLAG=
fi

echo
echo "Installing ITCL: $DIR $OSNAME $SH_LIB_EXT $BITS $TAR $SHARED_LIBS $MAKE_FLAGS"
echo

############
# Unpack:
	
install_source_dir=`pwd`
cd $DIR/src

if test -d $DIR/src/itcl3.1.0; then

    if test "$DO_NOT_UNTAR" != "yes"; then
        echo "It appears that iTcl has already been untarred.  Do you wish to untar it again? (y/n)"
        read answer
        echo
        if test "$answer" = "y"; then
            echo "Untar'ing... please wait."
            $TAR zxf $install_source_dir/Tarballs/itcl3.1.0.tar.gz
        fi
    fi
else
    echo "Untar'ing... please wait."
    $TAR zxf $install_source_dir/Tarballs/itcl3.1.0.tar.gz
fi

if test ! -L itcl; then
  ln -fs itcl3.1.0 itcl
fi

#  Make sure permissions are set such that we can change things if necessary.
echo "Updating permissions... please wait."
echo
chmod -R u+w itcl3.1.0

############
# Patch:

PATCH_FILE=$install_source_dir/Patches/$OSNAME'_'$BITS/itcl3.1.0.patch

if test -f $PATCH_FILE; then
    echo
    echo "#### PATCHING $OSNAME SPECIFIC ITCL/ITK FILES ####"
    echo
    cat $PATCH_FILE | patch -p0
    echo
    echo "#### DONE PATCHING FILES ####"
    echo
fi
   
############
# Configure: 

cd $DIR/src/itcl

./configure $SHARED_LIBS_FLAG --prefix=$DIR $BITS_FLAG

############
# Update Makefiles

if test "$BITS" = "64" && test "$OSNAME" = "IRIX64"; then 
    echo
    echo "#### FIXING MAKEFILE FOR 64bit SGI BUILD ####"
    echo

    for dir in itcl/unix itk/unix; do
        cd $dir
        mv Makefile Makefile.orig
        sed -e "s/-n32/-64/g" Makefile.orig > Makefile_s1
        sed -e "s/CFLAGS =/CFLAGS = -64 /g" Makefile_s1 > Makefile_s2
        # The following command is needed to create an RPATH for itk that includes itcl
        sed -e "s% -o lib% \\$\\(LD_SEARCH_FLAGS\\) -o lib%g" Makefile_s2 > Makefile
        cd ../..
    done
fi

############
# Make

$MAKE $MAKE_FLAGS
chmod u+x config/*
$MAKE install

############
# Clean up includes and add necessary symbolic links

cd $DIR
cp src/itcl/itcl/generic/itclDecls.h include
cp src/itcl/itcl/generic/itclIntDecls.h include
cp src/itcl/itcl/generic/itclInt.h include
cp src/itcl/itk/generic/itkDecls.h include

cd $DIR/lib
if test ! -L libitcl.$SH_LIB_EXT && test -e libitcl3.1.$SH_LIB_EXT; then
   ln -fs libitcl3.1.$SH_LIB_EXT libitcl.$SH_LIB_EXT
fi
if test ! -L libitk.$SH_LIB_EXT && test -e libitk3.1.$SH_LIB_EXT; then
   ln -fs libitk3.1.$SH_LIB_EXT libitk.$SH_LIB_EXT
fi
if test ! -L itcl && test -e itcl3.1; then
   ln -fs itcl3.1 itcl
fi
if test ! -L itk && test -e itk3.1; then
   ln -fs itk3.1 itk
fi

