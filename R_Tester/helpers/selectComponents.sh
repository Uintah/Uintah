#! /bin/bash

#
# This script asks the user (using a textual 'dialog' box) which RT
# components they want to test (eg: MPM, ARCHES, Wasatch, etc) and
# then prints out a string (to be consumed by another script) in the
# form of...
#
# -t COMP1 -t COMP2 [...]
#
# ...where COMP? => MPM, ICE, ARCHES, MPMARCHES, etc
#
# Note, if the environment variable TEST_COMPONENTS is set, then no
# dialog is displayed and the TEST_COMPONENTS value is displayed.
#

#
# Notes:
# 
# - The 'output' (stdout) of this script is grabbed by the script that calls
#   it, so (echo'd) messages are 'hidden' unless explicitly sent to "> /dev/stderr"... 
#   which needs to be done for all error messages.
#

if test ${TEST_COMPONENTS:+1}; then
  for comp in $TEST_COMPONENTS; do 
    echo "-t $comp "
  done
  exit 0
fi

if test "$TERM" == "dumb" -o "$TERM" == "emacs"; then
  echo > /dev/stderr
  echo "ERROR: `basename $0` requires a fully functional terminal... you have '$TERM'." > /dev/stderr
  echo "       (Consider setting environment variable TEST_COMPONENTS.)  Goodbye." > /dev/stderr
  echo > /dev/stderr
  exit 1
fi
if test "$EMACS" == "t"; then
  echo > /dev/stderr
  echo "ERROR: `basename $0` cannot be run from within emacs..." > /dev/stderr
  echo "       (Consider setting environment variable TEST_COMPONENTS.)  Goodbye." > /dev/stderr
  echo > /dev/stderr
  exit 1
fi

componentTests=`cd $1; ls *.py | grep -v __init__ | sed "s/.py//g" | sed "s,TestScripts/,,g"`

list=""
n="0"
for comp in $componentTests; do
  list="$list $comp - off,"
  n=$(( $n + 1 ))
done

# 'dialog' does not (natively) exist under OSX, so let user know...
if ! test `which dialog`; then
  echo "" > /dev/stderr
  echo "ERROR: the 'dialog' shell command not found.  (Use 'fink' to install under OSX.)" > /dev/stderr
  echo "       Please install 'dialog', or use the environment variable TEST_COMPONENTS." > /dev/stderr
  echo "" > /dev/stderr
  exit 1
fi


componentTest=`dialog --stdout --separate-output --checklist "Select the component for local regression testing" 20 61 15 $list`

if [ $? != 0 ] ; then
  echo "Cancel selected..." > /dev/stderr
  exit 1
fi 

for comp in $componentTest; do 
  echo "-t $comp "
done

exit 0
