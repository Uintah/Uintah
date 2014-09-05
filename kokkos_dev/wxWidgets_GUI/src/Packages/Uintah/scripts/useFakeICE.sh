#! /bin/bash

usage()
{
    echo
    echo "Usage:"
    echo
    echo "   $0 <path/to/top/of/scirun/<bin>>"
    echo
    echo "For example: $0 /home/user/SCIRun/sgi64opt"
    echo
    exit
}

if test $# != 1; then
    usage
fi

if test ! -d $1/include/sci_defs; then
    echo
    echo "Error: $1 does not appear to be a SCIRun binary tree... "
    echo "       (Specifically, the 'include/sci_defs/' sub-dir is missing.)"
    echo "       Please fix the directory parameter ($1) and run again."
    echo 
    exit
fi

#
# Make sure that the src dir is in the correct location...
#

if test ! -d $1/../src; then
    echo ""
    echo "ERROR: The src directory was not found at:"
    echo "       $1/../src"
    echo ""
    echo "       Contact dav@sci.utah.edu to udate this script"
    echo "       to handle this case.  Sorry for the inconvenience."
    echo ""
    exit
fi

# 
# Warn users about outdated lib files...
#

if test -f $1/lib/libPackages_Uintah_CCA_Components_Dummy.so \
     -o -f $1/lib/libPackages_Uintah_CCA_Components_Dummy.dylib; then
    echo ""
    echo "Warning:"
    echo ""
    echo "By running this script, it is possible that certain libraries"
    echo "and object files have become out of date.  Removing them now"
    echo "so that they will re-build correctly."
    echo ""
    echo "    rm -f $1/lib/libPackages_Uintah_CCA_Components_Dummy.*"
    echo "    rm -rf $1/Packages/Uintah/CCA/Components/Dummy"
    echo ""
    echo "!!!WARNING!!!"
    echo ""
    echo "If you have other <bin> directories (perhaps for other architectures)"
    echo "then their libPackages_Uintah_CCA_Components_Dummy.* and "
    echo "Packages/Uintah/CCA/Components/Dummy/ could also be out of date."
    echo "When you compile in those directories, you may get a message about"
    echo "missing or conflicting symbols.  If so, manually remove the above."
    echo ""

    rm -rf $1/Packages/Uintah/CCA/Components/Dummy
    rm -f $1/lib/libPackages_Uintah_CCA_Components_Dummy.*
fi

#
# Update Uintah/StandAlone/sub.mk
#

## Determine how many useFakeXXXXs are in place

filename=$1/../src/Packages/Uintah/StandAlone/sub.mk
using_fake_arches=`grep "#ARCHES_LIBS" $filename`
using_fake_mpmice=`grep "#MPMICE_LIB" $filename`
using_fake_ice=`grep "#ICE_LIB" $filename`

if test -z "$using_fake_ice"; then

    echo "Applying FakeICE to $filename..."

    # Handle Dummy lib if necessary...  (If it is already in file,
    # then the sed line will not do anything.)

    mv -f $filename $filename.tmp
    sed -e "s,#DUMMY_LIB,DUMMY_LIB ,g" \
        -e "s,^ICE_LIB ,#ICE_LIB,g" $filename.tmp > $filename
    rm $filename.tmp

    ############
    filename=$1/../src/Packages/Uintah/CCA/Components/sub.mk
    echo "Applying FakeICE to $filename..."
    mv -f $filename $filename.tmp
    sed -e "s,^#DUMMY_LIB,DUMMY_LIB ,g" \
        -e "s,^ICE ,#ICE,g" $filename.tmp > $filename
    rm $filename.tmp

    ############
    filename=$1/../src/Packages/Uintah/CCA/Components/Dummy/sub.mk
    echo "Applying FakeICE to $filename..."

    mv -f $filename $filename.tmp
    sed -e "s,#FAKE_ICE,FAKE_ICE ,g" $filename.tmp > $filename
    rm $filename.tmp

    ############
    filename=$1/../src/Packages/Uintah/CCA/Components/Parent/sub.mk
    echo "Applying FakeICE to $filename..."

    mv -f $filename $filename.tmp
    sed -e "s,^#DUMMY,DUMMY ,g" \
        -e "s,^ICE ,#ICE,g" $filename.tmp > $filename
    rm $filename.tmp

else # REVERSE FakeICE

    filename=$1/../src/Packages/Uintah/StandAlone/sub.mk
    echo "Reversing FakeICE from $filename..."

    if test -z "$using_fake_arches" -a -z "$using_fake_mpmice"; then
        sed_line1='-e "s,DUMMY_LIB ,#DUMMY_LIB,g"'
    fi

    mv -f $filename $filename.tmp
    sed_line2='-e "s,#ICE_LIB,ICE_LIB ,g"'
    eval sed $sed_line1 $sed_line2 $filename.tmp > $filename
    rm $filename.tmp

    ############
    filename=$1/../src/Packages/Uintah/CCA/Components/sub.mk
    echo "Reversing FakeICE from $filename..."

    sed_line2='-e "s,#ICE,ICE ,g"'

    mv -f $filename $filename.tmp
    eval sed $sed_line1 $sed_line2 $filename.tmp > $filename
    rm $filename.tmp

    ############
    filename=$1/../src/Packages/Uintah/CCA/Components/Dummy/sub.mk
    echo "Reversing FakeICE from $filename..."

    mv -f $filename $filename.tmp
    sed -e "s,FAKE_ICE ,#FAKE_ICE,g" $filename.tmp > $filename
    rm $filename.tmp

    ############
    filename=$1/../src/Packages/Uintah/CCA/Components/Parent/sub.mk
    echo "Reversing FakeICE from $filename..."

    if test -z "$using_fake_arches" -a -z "$using_fake_mpmice"; then
        sed_line1='-e "s,DUMMY ,#DUMMY,g"'
    fi

    mv -f $filename $filename.tmp
    sed_line2='-e "s,#ICE,ICE ,g"'
    eval sed $sed_line1 $sed_line2 $filename.tmp > $filename
    rm $filename.tmp
fi

