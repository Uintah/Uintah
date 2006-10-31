#!/bin/sh

do_pingpong=0
do_mxn=0
do_mem=0
do_parallel=0
do_serial=0

ALL=

usage () {
    echo
    echo "Usage: testscript [OPTION]..."
    echo "Run the tester.  With no OPTION, run all the tests."
    echo
    echo "  -all           run all tests (overrides other tests flags)"
    echo "  -pingpong      run the pingpong and pingpongArr tests"
    echo "  -mxn           run the LUFactor, Jacobi and OESort tests"
    echo "  -mem           run the memstress test"
    echo "  -parallel      run all parallel component tests"
    echo "  -serial        run all serial tests"
    echo;
    exit
}

for arg in $*; do

    case $arg in
	-all)
            ALL=yes
            ;;
	-pingpong)
	    echo; echo "Running the pingpong tests . . ."
	    do_pingpong=1;;
	-mxn)
	    echo; echo "Running the M-by-N tests . . ."
	    do_mxn=1;;
	-mem) 
	    echo; echo "Running the memory tests . . ."
	    do_mem=1;;
	-parallel)
	    echo; echo "Running all the parallel tests . . ."
	    do_parallel=1;;
	-serial)
	    echo; echo "Running all the serial tests . . ."
	    do_serial=1;;
	*)
	    usage ;;
    esac
done

# If no args, do all tests.
if test $# = 0; then
   ALL=yes
fi


run_simple_test ( ) {
    echo "----------------------------------"
    scim -t test/templatenone test/$1 
    if test $? -eq 0 
    then
      echo "testing " $1 " SUCCESSFUL"
    else 
      echo "testing " $1 " FAILED"
    fi
}

run_neg_test ( ) {
    echo "----------------------------------"
    scim -t test/templatenone test/$1
    if test $? -neq 0
    then
      echo "testing " $1 " SUCCESSFUL"
    else
      echo "testing " $1 " FAILED"
    fi
}

run_debug_diff_test ( ) {
    echo "----------------------------------"
    scim -d out.kwai -t test/$1 test/$2 ; diff test/$3 out.kwai 
    if test $? -eq 0
    then
      echo "testing " $1 " SUCCESSFUL"
    else
      echo "testing " $1 " FAILED"
    fi
}


run_error_diff_test ( ) {
    echo "----------------------------------"
    scim -t test/templatenone test/$1 | diff test/$2 -
    if test $? -eq 0
    then
      echo "testing " $1 " SUCCESSFUL"
    else
      echo "testing " $1 " FAILED"
    fi
}

run_diff_test ( ) {
    echo "----------------------------------"
    scim -t test/$1 -T $2 test/$3 ; diff test/$4 out.scim
    if test $? -eq 0
    then
      echo "testing " $1 " SUCCESSFUL"
    else
      echo "testing " $1 " FAILED"
    fi
}


if test -n "$ALL"; then
    echo; echo "Running all the tests..."
    echo "*********************************************************"
    run_simple_test testuno.sidl;
    run_simple_test testdue.sidl;
    run_simple_test testinclude.sidl;
    run_simple_test testmap.sidl;
    run_error_diff_test testmapnone.sidl testmapnone.error;
    run_error_diff_test testmapdouble.sidl testmapdouble.error; 
    run_debug_diff_test templatenone testmethodmap.sidl testmethodmap.out;
    run_diff_test testtemplateuno.erb heyhey testuno.sidl testtemplateuno.out;
    run_diff_test testtemplatedue.erb heyhey testnamemap.sidl testtemplatedue.out;
    run_diff_test testargs.erb heyhey testnm2.sidl testnm2.out;
    run_diff_test testtemplateuno.erb heyhey testomit.sidl testomit.out;
    run_simple_test basiccpp.h;
    run_diff_test testtemplateuno.erb heyhey parselang.h parselang-ttuno.out;
    echo "----------------------------------"
    exit
fi

# Actually do the tests

echo

if test $do_pingpong -eq 1; then
fi

if test $do_mxn -eq 1; then
fi

if test $do_mem -eq 1; then
fi

if test $do_parallel -eq 1; then
fi

if test $do_serial -eq 1; then
fi



