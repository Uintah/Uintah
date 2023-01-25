#! /bin/bash
export NCURSES_NO_UTF8_ACS=1   # fix for dialog issue with certain terminals

#  This script is called from generateGoldStandars.py and 
#  provides the user input.  It asks the user (using a
#  textual 'dialog' box) which RT components they want to 
#  test (eg: MPM, ARCHES, Wasatch, etc), the type of test to 
#  run (NIGHTLYTESTS, DEBUGTESTS, LOCALTESTS) and then prints 
#  out a string (to be consumed by another script) in the
#  form of...
#
# -t COMP1:TESTTYPE -t COMP2:TESTTYPE [...]
#
# ...where COMP? => MPM, ICE, ARCHES, MPMARCHES, etc
#          TESTTYPE? => NIGHTLYTESTS, DEBUGTESTS, LOCALTESTS [....]
#
# Note, if the environment variable TEST_COMPONENTS is set, then no
# dialog is displayed and the TEST_COMPONENTS:LOCALTESTS value is displayed.
#

#
# Notes:
# 
# - The 'output' (stdout) of this script is grabbed by the script that calls
#   it, so (echo'd) messages are 'hidden' unless explicitly sent to "> /dev/stderr"... 
#   which needs to be done for all error messages.
#

######################################################################3
#
# Find all the component tests (Each has a python script):
  
here="$1"

componentTests=`cd $here; ls *.py | grep -v __init__ | sed "s/.py//g" | sed "s,TestScripts/,,g"`


test_state_list=""
list=""
n="0"
for comp in $componentTests; do
  test_state_list="$test_state_list $comp - off,"
  list="$list $comp"
  n=$(( $n + 1 ))
done


######################################################################3
#
# If the environmental variable TEST_COMPONENTS is set, print them out and exit 
# 

theseTests=${WHICH_TESTS:="LOCALTESTS"}

if test ${TEST_COMPONENTS:+1}; then
  echo > /dev/stderr
  echo "Using TEST_COMPONENTS env var for list of components : ($TEST_COMPONENTS) and tests : ($theseTests)"> /dev/stderr
  echo > /dev/stderr

  for comp in $TEST_COMPONENTS; do 
    echo "-t $comp:$theseTests "
  done
  exit 0
fi

#__________________________________
# bulletproofing
if test "$TERM" == "dumb" -o "$TERM" == "emacs"; then
  echo > /dev/stderr
  echo "ERROR: `basename $0` requires a fully functional terminal... you have '$TERM'." > /dev/stderr
  echo "       Consider setting environment variable TEST_COMPONENTS to include:" > /dev/stderr
  echo "          $list" > /dev/stderr
  echo "       Goodbye." > /dev/stderr
  echo > /dev/stderr
  exit 1
fi
if test "$EMACS" == "t"; then
  echo > /dev/stderr
  echo "ERROR: `basename $0` cannot be run from within emacs..." > /dev/stderr
  echo "       Consider setting environment variable TEST_COMPONENTS to include:" > /dev/stderr
  echo "          $list" > /dev/stderr
  echo "       Goodbye." > /dev/stderr
  echo > /dev/stderr
  exit 1
fi


#__________________________________
# Have the user pick a component

# 'dialog' does not (natively) exist under OSX, so let user know...
if ! test `which dialog`; then
  echo "" > /dev/stderr
  echo "ERROR: the 'dialog' shell command not found.  (Use 'fink' to install under OSX.)" > /dev/stderr
  echo "       Please install 'dialog', or use the environment variable TEST_COMPONENTS." > /dev/stderr
  echo "" > /dev/stderr
  exit 1
fi


selectedCompTests=`dialog --stdout --separate-output --checklist "Select the component for local regression testing" 20 61 15 $test_state_list`

#remove quotations
selectedCompTests=`echo $selectedCompTests | tr -d '"'`


if [ $? != 0 ] ; then
  echo "Cancel selected..." > /dev/stderr
  exit 1
fi 

#__________________________________
# find what subset tests the user wants to run
let n=0
for componentTest in $selectedCompTests; do
  WHICH_TESTS[$n]="LOCALTESTS"  # default value

  let t=`grep -c LIST $here/$componentTest.py`
  testSubsets=`grep LIST $here/$componentTest.py | cut -d: -f2-10`

  #__________________________________
  # open dialog box if the component has the line
  #  "#LIST: LOCALTESTS NIGHTLYTESTS ......"

  if [ $t != 0 ]; then
    list=""
    for tests in $testSubsets; do
      if [ $tests == "LOCALTESTS" ]; then
        list="$list $tests - on"
      else
        list="$list $tests - off  "
      fi
    done

    declare -a selectedSubset=(`dialog --stdout --separate-output --checklist " [$componentTest] Select ONE set of tests to run" 20 61 15 $list`)

    #remove quotations from options variable
    selectedSubset=`echo $selectedSubset | tr -d '"'`

    if [ ${#selectedSubset[@]} == 0 ] ; then                                                                
      echo ""> /dev/stderr
      echo "Cancel selected... Goodbye." > /dev/stderr                                                                 
      echo ""
      exit 1                                                                                 
    fi
    if [ ${#selectedSubset[@]} != 1 ] ; then
      echo "ERROR: You selected more than one set of tests to run."> /dev/stderr
      echo "...now exiting"> /dev/stderr
      exit 1
    fi

    WHICH_TESTS[$n]=$selectedSubset

  fi
  let n=n+1
done


#__________________________________
# finally output the 
# -t COMP1:TESTTYPE -t COMP2:TESTTYPE

let n=0
for comp in $selectedCompTests; do
  test=${WHICH_TESTS[$n]}
  echo "-t $comp:$test "
  let n=n+1
done

exit 0
