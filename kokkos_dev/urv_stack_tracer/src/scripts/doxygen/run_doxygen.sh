#! /bin/sh
############################################################################
#
# run_doxygen.sh
#
# Author: J. Davison de St. Germain
# Date:   Mar 27, 2008
#
# Run's doxygen on the Uintah directory of the SCIRun source code tree.
# Places the output (as specified in the doxygen_config file) on the
# C-SAFE web site (http://www.csafe.utah.edu/dist/doxygen).  Updates
# the index.html file to include the SVN revision number of the tree.
#
# NOTE, run_doxygen.sh should be running nightly via a cron job.  See
# the README file for more information.
#
# WARNING: run_doxygen.sh runs an 'svn update' on the checked out code.
#
############################################################################

#
# WARNING: Make sure that DEST_DIR is consistent with the value
# specified in the doxygen_config file.
#
DEST_DIR=/usr/sci/projects/Uintah/www/dist/doxygen/uintah

function usage
{
    echo ""
    echo "Usage: run_doxygen.sh"
    echo ""
    echo "   Run from the src/ directory."
    echo ""
    exit
}

if test $# -gt 0; then
  usage
fi

if test ! -f /tmp/doing_dox; then
   touch /tmp/doing_dox
fi

echo ""              >> /tmp/doing_dox
echo "START: `date`" >> /tmp/doing_dox
echo "START: `date`"

########
#
# Verify that script is running from the correct directory:
#

PWD=`pwd`
cur_dir=`basename $PWD`
temp=`dirname $PWD`
par_dir=`basename $temp`
temp=`dirname $temp`
gpar_dir=`basename $temp`

if test $cur_dir != "src"; then
   echo
   echo "Error: This script must be run from the src directory in the"
   echo "       Uintah src tree.  Exiting."
   echo
   exit
fi
if test ! -f scripts/doxygen/doxygen_config; then
   echo
   echo "Error: doxgen_config file not found in scripts/doxygen directory.  Exiting."
   echo
   exit
fi
#
########

########
#
# Update the SVN tree
#
echo "Updating SVN repository..."
svn update
#
########

# Verify doxygen exists
which doxygen > /dev/null 2>&1

if test $? != 0; then
   echo
   echo "Error: doxgen executable not found in path.  Exiting."
   echo
   exit
fi

# Back up current doxygen output.  We save one previous day's output.
# (Have to move it up one directory or the doxygen command deletes it.
#
if test -d $DEST_DIR/html; then
   echo ""
   echo "Backing up current (yesterday's) doxygen output..."
   echo ""
   if test ! -d $DEST_DIR/../uintah-yesterday; then
      mkdir $DEST_DIR/../uintah-yesterday
   fi

   rm -rf $DEST_DIR/../uintah-yesterday/html
   mv $DEST_DIR/html $DEST_DIR/../uintah-yesterday/html
fi

#### DEBUG:
##echo "here:"
##ls -l $DEST_DIR
##echo "now run doxygen"
doxygen scripts/doxygen/doxygen_config > /dev/null

#### DEBUG:
##echo "doxygen is done:"
##ls -l $DEST_DIR
##echo "moving on"

########
#
# Add the SVN revision number to the HTML (using 'sed')...
#
CODE_REVISON=`svn info | grep Revision`

echo "Code svn version is $CODE_REVISON"

mv $DEST_DIR/html/index.html $DEST_DIR/html/index.html.orig
sed "s,Uintah,Uintah ($CODE_REVISON),g" $DEST_DIR/html/index.html.orig > $DEST_DIR/html/index.html
rm $DEST_DIR/html/index.html.orig
#
########

# Make sure everyone has permission to see the files...
chmod -R go+rX $DEST_DIR

echo "END:   `date`" >> /tmp/doing_dox
echo "END:   `date`"
