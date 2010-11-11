#! /bin/sh

#
# This script asks the user which RT components they want to test (eg: MPM, ARCHES, Wasatch, etc)
# and then prints out a string (to be consumed by another script) with those components.
#

if test "$TERM" == "dumb" -o "$TERM" == "emacs"; then
  echo > /dev/stderr
  echo "ERROR: `basename $0` requires a fully functional terminal... you have '$TERM'.  Goodbye." > /dev/stderr
  echo > /dev/stderr
  exit 1
fi
if test "$EMACS" == "t"; then
  echo > /dev/stderr
  echo "ERROR: `basename $0` cannot be run from within emacs...  Goodbye." > /dev/stderr
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

componentTest=`dialog --stdout --separate-output --checklist "Select the component for local regression testing" 20 61 15 $list`
if [ $? != 0 ] ; then
  echo "Cancel selected..." > /dev/stderr
  exit 1
fi 

for comp in $componentTest; do 
  echo "-t $comp "
done

exit 0
