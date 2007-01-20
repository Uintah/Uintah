#! /bin/sh

#
#  make_tarball.sh
#
#  Author: J. Davison de St. Germain 
#
#  Run this script from the the directory above the Thirdparty_install
#  directory.
#
#  This script creates the Thirdparty_install.<version>.tar.gz file
#
#

################################################################
# Usage Function

show_usage ()
{
    echo ""
    echo "Usage: $0"
    echo ""
    echo "  This script must be run from the directory above the 'Thirdparty'"
    echo "  directory.  It tars that directory into Thirdparty.<version>.tar.gz"
    echo ""
    exit
}

################################################################
# Test parameters

if test $# != 0; then
    echo ""
    echo "ERROR: Bad number of parameters."
    show_usage $0
fi

if test ! -f Thirdparty/install.sh; then
    echo ""
    echo "ERROR: Cannot find Thirdparty/install.sh..."
    echo "       You most likely running this script from the wrong location."
    echo "       It is also possible that your 'Thirdparty' directory is invalid."
    show_usage $0
fi

################################################################
# Move and ln files.

PWD=`pwd`

echo 
echo "Getting Thirdparty version from: Thirdparty/install.sh"
TP_VER=`grep "tp_ver=" Thirdparty/install.sh  | cut -f2 -d"="  | cut -f2 -d"'" `

if test "$TP_VER" = ""; then
   echo
   echo "ERROR: Unable to determine the Thirdparty version."
   echo
   exit
else
   echo
   echo "Version is $TP_VER."
   echo
fi

######################################################################
# Make sure that we are starting clean:

if test -d "Thirdparty.$TP_VER"; then
   echo "Warning: Thirdparty.$TP_VER already exists!" 
   echo
   echo "Delete it? (y/n)"
   read answer

   if test "$answer" = "y"; then
       echo
       echo "Deleting..."
       rm -rf  Thirdparty.$TP_VER
       echo
   else
       echo
       echo "Aborting. Bye."
       echo
       exit
   fi
fi

if test -f "Thirdparty.$TP_VER.tar.gz"; then
   echo "Warning: Thirdparty.$TP_VER.tar.gz already exists!" 
   echo
   echo "Delete it? (y/n)"
   read answer

   if test "$answer" = "y"; then
       echo
       echo "Deleting..."
       rm -f  Thirdparty.$TP_VER.tar.gz
       echo
   else
       echo
       echo "Aborting. Bye."
       echo
       exit
   fi
fi

######################################################################
# Make the tarball.

echo "Creating Thirdparty.$TP_VER.tar.gz"
echo

echo "...copying Thirdparty to temporary directory..."
echo

mkdir Thirdparty.$TP_VER
cp -rf Thirdparty/* Thirdparty.$TP_VER

# We don't send the 'Develop' or '.svn' directories.
rm -f Thirdparty.$TP_VER/install_command.txt
rm -rf Thirdparty.$TP_VER/Develop
find Thirdparty.$TP_VER -name ".svn" -type d | xargs rm -rf

echo "- Would you like me to make sure there are no ~ (backup) files?  (y/n)"
read answer

if test "$answer" = "y"; then
    echo
    echo "Deleting these backup files:"
    echo
    find Thirdparty.$TP_VER -name "*~*" 
    find Thirdparty.$TP_VER -name "*~*" | xargs rm -f
    echo
fi

# Update permissions
chmod -R go+rX Thirdparty.$TP_VER

chmod ugo+x Thirdparty.$TP_VER/install.sh
chgrp -R sci Thirdparty.$TP_VER

# Create tarball
echo "Creating Tarball..."
echo ""
tar -zcvf Thirdparty.$TP_VER.tar.gz Thirdparty.$TP_VER

# Change permissions
echo 
echo "Updating permissions (go+r) on tarball..."
echo 
chmod go+r Thirdparty.$TP_VER.tar.gz

######################################################################
# Clean up

echo "Cleaning up..."
echo ""
rm -rf Thirdparty.$TP_VER
echo
echo "Done."

echo
echo "Things you might want to do:"
echo
echo "mv Thirdparty.$TP_VER.tar.gz /usr/sci/projects/SCIRun/Thirdparty/dist"
echo "chmod go+r,g+w /usr/sci/projects/SCIRun/Thirdparty/dist/Thirdparty.$TP_VER.tar.gz"
echo
echo "Upload the Thirdparty tarball to the Code Wiki:"
echo "    https://code.sci.utah.edu:443/wiki/index.php/SCIRun_Thirdparty"
echo