##  ---------------------------------------------------------------------
##  --------------------  new macro definitions  ------------------------
##  ---------------------------------------------------------------------

##  none of the new macros can be nested with or within other macros

##  these are platform sensitive - be very careful, when adding or
##  editing, to make sure it works on all required platforms

##  SCI_MSG_ERROR(message)
##
##     Wrapper for AC_MSG_ERROR but tells user to try configuring with
##     --enable-verbosity to get more help in figuring out what is
##     wrong.

AC_DEFUN([SCI_MSG_ERROR], [
## SCI_MSG_ERROR
if test -z "$debugging"; then
  AC_MSG_WARN([
    Adding '--enable-verbosity' to configure line
    _may_ give more information about what is breaking.
    IF YOU KNOW WHAT YOU ARE DOING, TRY THIS:
    Digging through the config.log file is your best option
    for determining what went wrong.  Search for the specific lib/file
    that configured failed to find.  There should be a compile line
    and source code near that check.  If you cut the source code into
    a test.cc and then use the compile line to compile it, the true
    error most likely will show up.
  ])
fi

AC_MSG_ERROR($1)
])

AC_DEFUN([SCI_MSG_DEBUG], [
if test "$debugging" = "yes"; then
  AC_MSG_NOTICE([
debug info:
$1
  ])
fi
])

AC_DEFUN([SCI_TRY_LINK], [
## arguments mean:
## arg 1 : variable base name i.e. MATH
## arg 2 : checking message
## arg 3 : includes that arg 6 needs to compile
## arg 4 : include paths -I
## arg 5 : list of libs to link against -l
## arg 6 : lib paths -L
## arg 7 : extra link flags 
## arg 8 : body of code to compile. can be empty
## arg 9 : 'optional' or 'required' or 'specific' required argument
##             If specific, then SCI_TRY_LINK will take only one lib
##             path and one include path and will verify that the
##             libs/includes are in that path.
## 
## after execution of macro, the following will be defined:
##      Variable            Value
##      --------            -----
##      LIB_DIR_$1        => lib path
##      $1_LIB_DIR_FLAG   => all the -L's
##      $1_LIB_FLAG       => all the -l's
##      HAVE_$1           => yes or no
##      INC_$1_H          => all the -I's
##      HAVE_$1_H         => yes or no

if test $# != 9; then
     AC_MSG_ERROR(Wrong number of parameters ($#) for SCI-TRY-LINK for $2.  This is an internal SCIRun configure error and should be reported to scirun-develop@sci.utah.edu.)
fi


ifelse([$1],[],[AC_FATAL(must provide a test name in arg 1)],)dnl

AC_MSG_CHECKING(for $2 ($9))
_sci_savelibs=$LIBS
_sci_saveldflags=$LDFLAGS
_sci_savecflags=$CFLAGS
_sci_savecxxflags=$CXXFLAGS

_sci_includes=

if test "$9" != "optional" -a "$9" != "required" -a "$9" != "specific"; then
     AC_MSG_ERROR(Last parameter of SCI-TRY-LINK for $2 must be: optional or required or specific.  (You had $9.)  This is an internal SCIRun configure error and should be reported to scirun-develop@sci.utah.edu.)
fi

# If $4 (the -I paths) is blank, do nothing, else do the for statement.
ifelse([$4],[],,[
for i in $4; do
  if test "$i" = "/usr/include"; then
     echo ""
     AC_MSG_ERROR(Please do not specify /usr/include as the location for $2 include files.)
  fi
  # make sure it exists
  if test -d $i; then
    if test -z "$_sci_includes"; then
      _sci_includes=-I$i
    else
      _sci_includes="$_sci_includes -I$i"
    fi
  fi
done
])dnl

if test "$9" = "specific"; then
  # If 'specific' then only one lib is allowed:
  ### Determine if there is only one item in $6 (I don't know of a better way to do this.)
  __sci_pass=false
  __sci_first_time=true

  # Must have the "" for the SGI sh.
  for value in "" $6; do
    if test "$value" = "/usr/lib"; then
       echo ""
       AC_MSG_ERROR(Please do not specify /usr/lib as the location for libs $5.)
    fi
    if test "$value" = ""; then
      continue
    fi
    if test $__sci_first_time = "true"; then  
      __sci_first_time=false
      __sci_pass=true
    else
      __sci_pass=false
    fi
  done
  if test $__sci_pass != "true"; then
       AC_MSG_ERROR(For specific SCI-TRY-LINK test for $2 only one library path may be specified (you had: $6).  This is an internal SCIRun configure error and should be reported to scirun-develop@sci.utah.edu.)
  fi
  # and only one include path:
  ### Determine if there is only one item in $4
  __sci_pass=false
  __sci_first_time=true
  for value in ""$4; do
    if test "$value" = ""; then
      continue
    fi
    if test $__sci_first_time = "true"; then  
      __sci_first_time=false
      __sci_pass=true
    else
      __sci_pass=false
    fi
  done
  if test $__sci_pass != "true"; then
       AC_MSG_ERROR(For specific SCI-TRY-LINK test for $2 only one include path may be specified (you had: $4).  This is an internal SCIRun configure error and should be reported to scirun-develop@sci.utah.edu.)
  fi
fi
  
ifelse([$5],[],_sci_libs=,[
for i in $5; do
  if test -z "$_sci_libs"; then
    _sci_libs=-l$i
  else
    _sci_libs="$_sci_libs -l$i"
  fi
done
])dnl

_sci_lib_path=
ifelse([$6],[],,[
for i in $6; do
  # make sure it exists
  if test -d $i; then
    if test -z "$_sci_lib_path"; then
      _sci_lib_path="$LDRUN_PREFIX$i -L$i"
    else
      _sci_lib_path="$_sci_lib_path $LDRUN_PREFIX$i -L$i"
    fi
  fi
done
])dnl

CFLAGS="$_sci_includes $CFLAGS"
CXXFLAGS="$_sci_includes $CXXFLAGS"
LDFLAGS="$_sci_lib_path $LDFLAGS"
LIBS="$_sci_libs $7 $LIBS"

# Build up a list of the #include <file> lines for use in compilation:
__sci_pound_includes=
for i in ""$3; do
    # Have to have the "" for the SGI sh. 
    if test "$i" = ""; then
      continue
    fi
    __sci_pound_includes="$__sci_pound_includes
#include <$i>"
done

AC_TRY_LINK($__sci_pound_includes,[$8],[
eval LIB_DIR_$1='"$6"'

if test "$6" = "$SCI_THIRDPARTY_LIB_DIR"; then
  eval $1_LIB_DIR_FLAG=''
else
  eval $1_LIB_DIR_FLAG='"$_sci_lib_path"'
fi

eval $1_LIB_FLAG='"$_sci_libs"'
eval HAVE_$1="yes"

if test "$_sci_includes" = "$INC_SCI_THIRDPARTY_H"; then
  eval INC_$1_H=''
else
  eval INC_$1_H='"$_sci_includes"'
fi

eval HAVE_$1_H="yes"
AC_MSG_RESULT(yes)
], 
[
eval LIB_DIR_$1=''
eval $1_LIB_DIR_FLAG=''
eval $1_LIB_FLAG=''
eval HAVE_$1="no"
eval INC_$1_H=''
eval HAVE_$1_H="no"
AC_MSG_RESULT(not found)
if test "$9" != "optional"; then
  SCI_MSG_ERROR([[Test for required $1 failed. 
    To see the failed compile information, look in config.log, 
    search for $1. Please install the relevant libraries
     or specify the correct paths and try to configure again.]])
fi
])

if test "$9" = "specific"; then
#echo specific
  # Make sure the exact includes were found
  for i in ""$3; do
#echo looking for $4/$i
    if test ! -e $4/$i; then
     AC_MSG_ERROR(Specificly requested $2 include file '$4/$i' was not found)
    fi
  done
  # Make sure the exact libraries were found
  for i in ""$5; do
#echo looking for $6/$i
    if test ! -e $6/lib$i.so && test ! -e $6/lib$i.a; then
     AC_MSG_ERROR(Specificly requested $2 library file '$6/$i' was not found)
    fi
  done
fi


#restore variables
CFLAGS=$_sci_savecflags
CXXFLAGS=$_sci_savecxxflags
LDFLAGS=$_sci_saveldflags
LIBS=$_sci_savelibs
_sci_lib_path=''
_sci_libs=''
_sci_includes=''

])

##
##  SCI_CHECK_VERSION(prog,verflag,need-version,if-correct,if-not-correct,comp)
##
##  check whether the prog's version is >= need-version.
##  currently only supports version numbers of the form NUM(.NUM)*, no
##  letters allowed!  comp is the optional comparison used to _reject_
##  the program's version - the default is "-gt" (i.e. is need > have)
##

AC_DEFUN(SCI_CHECK_VERSION,
  [
    ##  SCI_CHECK_VERSION
    _SCI_CORRECT_='echo $echo_n "$echo_c"'
    _SCI_NOTCORRECT_='echo $echo_n "$echo_c"'
    _SCI_VER_1_="0"
    _SCI_VER_2_="$3"
    _CUR_1_=""
    _CUR_2_=""

    AC_MSG_CHECKING(for `basename $1` version $3)

    if test "$4"; then
      _SCI_CORRECT_='$4'
    fi

    if test "$5"; then
      _SCI_NOTCORRECT_='$5'
    fi

    if test "$6"; then
      _SCI_COMP_="$6"
    else
      _SCI_COMP_="-gt"
    fi

    eval "$1 $2 2> conftest.out >> conftest.out"
    _SCI_REPORT_="`cat conftest.out | head -n 1 | sed 's%[[^0-9\. ]]*%%g;s%^[ ]*%%' | cut -f1 -d' ' `"
    _SCI_VER_1_="$_SCI_REPORT_"

    _SCI_BIGGER_=yes
    _SCI_LAST_=""
    while test "$_SCI_VER_2_"; do
      if test "$_SCI_LAST_" = "$_SCI_VER_2_"; then
        break
      fi
      _SCI_LAST_="$_SCI_VER_2_"
      _CUR_1_=`echo $_SCI_VER_1_ | sed 's%\.[[0-9]]*[[a-z]]*%%g'`
      _SCI_VER_1_=`echo $_SCI_VER_1_ | sed 's%[[0-9]]*[[a-z]]*\.%%'`
      _CUR_2_=`echo $_SCI_VER_2_ | sed 's%\.[[0-9]]*[[a-z]]*%%g'`
      _SCI_VER_2_=`echo $_SCI_VER_2_ | sed 's%[[0-9]]*[[a-z]]*\.%%'`
      if test $_CUR_2_ $_SCI_COMP_ $_CUR_1_; then
        _SCI_BIGGER_=no
        break
      elif test $_CUR_1_ -gt $_CUR_2_; then
        break
      fi
    done

    if test "$_SCI_BIGGER_" = "yes"; then
      AC_MSG_RESULT(yes ($_SCI_REPORT_))
      eval $_SCI_CORRECT_
    else
      AC_MSG_RESULT(no ($_SCI_REPORT_))
      eval $_SCI_NOTCORRECT_
    fi
  ])

##
##  SCI_CHECK_VAR_VERSION(name,var,need-version,if-correct,if-not-correct,comp)
##
##  check whether the var (which represents a version number) version
##  is >= need-version.
##  currently only supports version numbers of the form NUM(.NUM)*, no
##  letters allowed!  comp is the optional comparison used to _reject_
##  the program's version - the default is "-gt" (i.e. is need > have)
##

AC_DEFUN(SCI_CHECK_VAR_VERSION,
  [
    ##  SCI_CHECK_VAR_VERSION
    _SCI_CORRECT_='echo $echo_n "$echo_c"'
    _SCI_NOTCORRECT_='echo $echo_n "$echo_c"'
    _SCI_VER_1_="0"
    _SCI_VER_2_="$3"
    _CUR_1_=""
    _CUR_2_=""

    AC_MSG_CHECKING(for $1 version $3)

    if test "$4"; then
      _SCI_CORRECT_='$4'
    fi

    if test "$5"; then
      _SCI_NOTCORRECT_='$5'
    fi

    if test "$6"; then
      _SCI_COMP_="$6"
    else
      _SCI_COMP_="-gt"
    fi

    eval "echo $2 2> conftest.out >> conftest.out"
    _SCI_REPORT_="`cat conftest.out | head -n 1 | sed 's%[[^0-9\.]]*%%g'`"
    _SCI_VER_1_="$_SCI_REPORT_"

    _SCI_BIGGER_=yes
    _SCI_LAST_=""
    while test "$_SCI_VER_2_"; do
      if test "$_SCI_LAST_" = "$_SCI_VER_2_"; then
        break
      fi
      _SCI_LAST_="$_SCI_VER_2_"
      _CUR_1_=`echo $_SCI_VER_1_ | sed 's%\.[[0-9]]*[[a-z]]*%%g'`
      _SCI_VER_1_=`echo $_SCI_VER_1_ | sed 's%[[0-9]]*[[a-z]]*\.%%'`
      _CUR_2_=`echo $_SCI_VER_2_ | sed 's%\.[[0-9]]*[[a-z]]*%%g'`
      _SCI_VER_2_=`echo $_SCI_VER_2_ | sed 's%[[0-9]]*[[a-z]]*\.%%'`
      if test $_CUR_2_ $_SCI_COMP_ $_CUR_1_; then
        _SCI_BIGGER_=no
        break
      elif test $_CUR_1_ -gt $_CUR_2_; then
        break
      fi
    done

    if test "$_SCI_BIGGER_" = "yes"; then
      AC_MSG_RESULT(yes ($_SCI_REPORT_))
      eval $_SCI_CORRECT_
    else
      AC_MSG_RESULT(no ($_SCI_REPORT_))
      eval $_SCI_NOTCORRECT_
    fi
  ])

##
##  SCI_CHECK_OS_VERSION(need-version,if-correct,if-not-correct,comp)
##
##  check whether the OS's version (uname -r) is >= need-version.
##  currently only supports version numbers of the form NUM(.NUM)*, no
##  letters allowed!  comp is the optional comparison used to _reject_
##  the program's version - the default is "-gt" (i.e. is need > have)
##

AC_DEFUN(SCI_CHECK_OS_VERSION,
  [
    ##  SCI_CHECK_OS_VERSION

    _SCI_CORRECT_='echo $echo_n "$echo_c"'
    _SCI_NOTCORRECT_='echo $echo_n "$echo_c"'
    _SCI_VER_1_="0"
    _SCI_VER_2_="$1"
    _CUR_1_=""
    _CUR_2_=""

    AC_MSG_CHECKING(for OS version $1)

    if test "$2"; then
      _SCI_CORRECT_='$2'
    fi

    if test "$3"; then
      _SCI_NOTCORRECT_='$3'
    fi

    if test "$6"; then
      _SCI_COMP_="$6"
    else
      _SCI_COMP_="-gt"
    fi

    eval "uname -r 2> conftest.out >> conftest.out"
    _SCI_REPORT_="`cat conftest.out | head -n 1 | sed 's%[[^0-9\.]]*%%g'`"
    _SCI_VER_1_="$_SCI_REPORT_"

    _SCI_BIGGER_=yes
    _SCI_LAST_=""
    while test "$_SCI_VER_2_"; do
      if test "$_SCI_LAST_" = "$_SCI_VER_2_"; then
        break
      fi
      _SCI_LAST_="$_SCI_VER_2_"
      _CUR_1_=`echo $_SCI_VER_1_ | sed 's%\.[[0-9]]*[[a-z]]*%%g'`
      _SCI_VER_1_=`echo $_SCI_VER_1_ | sed 's%[[0-9]]*[[a-z]]*\.%%'`
      _CUR_2_=`echo $_SCI_VER_2_ | sed 's%\.[[0-9]]*[[a-z]]*%%g'`
      _SCI_VER_2_=`echo $_SCI_VER_2_ | sed 's%[[0-9]]*[[a-z]]*\.%%'`
      if test $_CUR_2_ $_SCI_COMP_ $_CUR_1_; then
        _SCI_BIGGER_=no
        break
      elif test $_CUR_1_ -gt $_CUR_2_; then
        break
      fi
    done

    if test "$_SCI_BIGGER_" = "yes"; then
      AC_MSG_RESULT(yes ($_SCI_REPORT_))
      eval $_SCI_CORRECT_
    else
      AC_MSG_RESULT(no ($_SCI_REPORT_))
      eval $_SCI_NOTCORRECT_
    fi
  ])

##
##  SCI_ARG_WITH(arg-string, usage-string, if-used, if-not-used)
##
##    if an arg is provide to the "with", the arg must be a directory
##    or a file.  If not, a configure error is raised.  This will avoid
##    the problem of mis-typing the name of the "--with" dir/file.
##
##    does the same thing as AC_ARG_WITH (infact, it uses it), but
##    also appends arg-string to the master list of arg-strings
##

AC_DEFUN([SCI_ARG_WITH], [
  AC_ARG_WITH($1, $2, $3, $4)
  sci_arg_with_list="$sci_arg_with_list --with-$1 --without-$1"
  if test -n "$$1" -a ! -e "$$1"; then 
    AC_MSG_ERROR(The file or directory parameter ($$1) specified for --with-$1 does not exist!  Please verify that the path and file are correct.)
  fi

])

##
##  SCI_ARG_ENABLE(arg-string, usage-string, if-used, if-not-used)
##
##  does the same thing as AC_ARG_ENABLE (infact, it uses it), but
##  also appends arg-string to the master list of arg-strings
##

AC_DEFUN([SCI_ARG_ENABLE], [
  AC_ARG_ENABLE($1, $2, $3, $4) 
  sci_arg_enable_list="$sci_arg_enable_list --enable-$1 --disable-$1"
])

##
## SCI_ARG_VAR
##
## callse AC_ARG_VAR makes variables precious, and allows the vars to pass
## our valid check by recording it in sci_arg_var_list
##

AC_DEFUN([SCI_ARG_VAR], [
  AC_ARG_VAR($1, $2)
  sci_arg_var_list="$sci_arg_var_list $1"
])

##
##  INIT_PACKAGE_CHECK_VARS
##  
##  Initialize all the variables that guard REQUIRED dependencies
##  by specific configurations.
##
AC_DEFUN([INIT_PACKAGE_CHECK_VARS], [

  # This list is alphabetical.  Please keep it that way.
  sci_required_audio=no
  sci_required_awk=no
  sci_required_babel=no
  sci_required_blas=no
  sci_required_crypto=no
  sci_required_etags=no
  sci_required_exc=no 
  sci_required_fortran=no
  sci_required_hdf5=no
  sci_required_globus=no
  sci_required_glui=no
  sci_required_glut=no
  sci_required_gmake=no 
  sci_required_gzopen=no
  sci_required_hypre=no
  sci_required_insight=no
  sci_required_jpeg=no
  sci_required_lapack=no
  sci_required_mdsplus=no
  sci_required_mpi=no
  sci_required_netsolve=no
  sci_required_oogl=no
  sci_required_perl=no
  sci_required_petsc=no
  sci_required_plplot=no
  sci_required_qt=no
  sci_required_ssl=no
  sci_required_tau=no
  sci_required_teem=no
  sci_required_thirdparty=no
  sci_required_tiff=no
  sci_required_tools=no
  sci_required_unipetc=no
  sci_required_uuid=no
  sci_required_vdt=no

])
##
##  SCI_SET_PACKAGE_CHECKS
##  $1 is the name of a package.
##  
##  Set the variables that enable configure checks required for the Package.
##
AC_DEFUN([SCI_SET_PACKAGE_CHECKS], [

eval pkg_$1=yes

case $1 in
  BioPSE)
  ;;
  Teem)
    sci_required_teem=yes
  ;;
  VDT)
    sci_required_vdt=yes
  ;;
  MatlabInterface)
  ;;
  Uintah)
    sci_required_fortran=yes
    sci_required_mpi=yes
    sci_required_blas=yes
    sci_required_lapack=yes
    sci_required_perl=yes
    sci_required_tools=yes
  ;;
  Fusion)
  ;;
  DataIO)
    sci_required_mdsplus=yes
    sci_required_hdf5=yes
  ;;
  SCIRun2)
    sci_required_babel=yes
    sci_required_uuid=yes
  ;;
  Remote)
    sci_required_jpeg=yes
    sci_required_tiff=yes
  ;;
  NetSolve)
    sci_required_netsolve=yes
  ;;
  rtrt)
    sci_required_glut=yes
    sci_required_glui=yes
    sci_required_oogl=yes
    sci_required_audio=yes 
    sci_required_teem=yes
  ;;
  Insight)
    sci_required_insight=yes
    sci_required_xalan=yes
  ;;
  *)
    AC_MSG_WARN(No known dependencies for Package $1)
  ;;
esac

])
