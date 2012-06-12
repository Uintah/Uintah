##
##  For more information, please see: http://software.sci.utah.edu
## 
##  The MIT License
## 
##  Copyright (c) 2004 Scientific Computing and Imaging Institute,
##  University of Utah.
## 
##  License for the specific language governing rights and limitations under
##  Permission is hereby granted, free of charge, to any person obtaining a
##  copy of this software and associated documentation files (the "Software"),
##  to deal in the Software without restriction, including without limitation
##  the rights to use, copy, modify, merge, publish, distribute, sublicense,
##  and/or sell copies of the Software, and to permit persons to whom the
##  Software is furnished to do so, subject to the following conditions:
## 
##  The above copyright notice and this permission notice shall be included
##  in all copies or substantial portions of the Software.
## 
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
##  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
##  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
##  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
##  DEALINGS IN THE SOFTWARE.
##

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
    IF YOU KNOW WHAT YOU ARE DOING - TRY THIS:
    Digging through the config.log file is your best option
    for determining what went wrong.  Search for the specific lib/file
    that configured failed to find.  There should be a compile line
    and source code near that check.  If you cut the source code into
    a test.cc and then use the compile line to compile it - the true
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

AC_DEFUN([BASE_LIB_PATH], [
##
## BASE_LIB_PATH:
##
## This macro takes the base path of a library and determines
## if there is a lib32, lib64, or just plain lib that is 
## associated with it.  It sets the full path to the value of
## the first argument.
##
## eg:  if arg 2 is /usr/sci/local, and the machine has a 
## /usr/sci/local/lib64, and configure is in 64 bit mode, then
## /sur/sci/local/lib64 is set as the value of arg 1.
##
## arguments mean:
## arg 1 : This argument will be written with the result
## arg 2 : library base.  I.e. /usr, /usr/local, /usr/X11R6
##
## This function assums the enable_64bit variable is in scope
##

  _new_lib_path=$2/lib

  ## if 64 bit is enabled, check for lib64 first, otherwise use lib
  if test "$enable_64bit" = "yes"; then
    _new_lib_path=$2/lib64
    if test ! -e $_new_lib_path; then
      _new_lib_path=$2/lib
    fi
  else
  ## We should look for lib32 (IRIX), then default to lib
    _new_lib_path=$2/lib32
    if test ! -e $_new_lib_path; then
      _new_lib_path=$2/lib
    fi
  fi
  eval $1='"$_new_lib_path"'
])


AC_DEFUN([ABSOLUTE_LIBRARY], [
##
## ABSOLUTE_LIBRARY:
##
## arguments mean:
##
## arg 1 : This argument will be written with the result
## arg 2 : library flag (should be -L/some/dir -lsomelib
##
## The result (which the varaible named by arg1 is set to)
## will be /some/dir/somelib.<ext>
##

  _final_libs=""

echo "starting with $2"

  # Remove any -Wl's
  _lib_list=`echo "$2"      | sed "s%-Wl[[.,-/a-zA-Z0-9]]*%%g"`

echo "_lib_list: $_lib_list"
  # Find all the -L's and -l's.

  _BigLs=`echo $_lib_list   | sed "s%-l[[./a-zA-Z0-9]]*%%g"`
  _SmallLs=`echo $_lib_list | sed "s%-L[[./a-zA-Z0-9]]*%%g"`

  # Must add /usr/lib at the end so that libraries in /usr/lib
  # can be found.  
  _BigLs="$_BigLs /usr/lib"

echo "BigLs: $_BigLs"
echo "SmlLs: $_SmallLs"

  _checked_dirs=""

  for _bigL in $_BigLs; do

     _the_dir=`echo $_bigL | sed "s%-L%%"`
echo "the_dir: $_the_dir"

     _already_checked=`echo "$_checked_dirs" | grep "$_the_dir"`
     if test -n "$_already_checked"; then
        echo "Already checked in $_the_dir.  Skipping."
        continue
     fi

     for _smallL in $_SmallLs; do
        _the_lib=`echo $_smallL | sed "s%-l%%"`

echo "the_lib: $_the_lib"
        for _extension in so dylib a; do
           _lib_name="$_the_dir/lib$_the_lib.$_extension"

           if test -e $_lib_name; then
              _checked_dirs="$_checked_dirs $_the_dir"
echo "found it with .$_extension"
              _final_libs="$_final_libs $_lib_name"
              break
           fi
        done
     done
  done

  eval $1='"$_final_libs"'
])

## SCI_COMPILE_LINK_TEST (was the original SCI_TRY_LINK...)
AC_DEFUN([SCI_COMPILE_LINK_TEST], [
## arguments mean:
## arg 1 : variable base name i.e. MATH
## arg 2 : checking message
## arg 3 : includes that arg 6 needs to compile
## arg 4 : include paths -I
## arg 5 : list of libs to link against -l
## arg 6 : lib paths -L
## arg 7 : extra link flags 
## arg 8 : body of code to compile. can be empty
## arg 9 : optional or not-optional required argument
## 
## after execution of macro, the following will be defined:
##      Variable            Value
##      --------            -----
##      $1_LIB_DIR        => lib path
##      $1_LIB_DIR_FLAG   => all the -L's
##      $1_LIB_FLAG       => all the -l's
##      HAVE_$1           => yes or no
##      INC_$1_H          => all the -I's
##      HAVE_$1_H         => yes or no

ifelse([$1],[],[AC_FATAL(must provide a test name in arg 1)],)dnl

ifelse([$9],[optional],,[$9],[not-optional],,
       [AC_FATAL(arg 9 must be either 'optional' or 'not-optional' got $9)])dnl

AC_MSG_CHECKING(for $2 ($9))
_sci_savelibs=$LIBS
_sci_saveldflags=$LDFLAGS
_sci_savecflags=$CFLAGS
_sci_savecxxflags=$CXXFLAGS

_sci_includes=
ifelse([$4],[],,[
for i in $4; do
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

AC_TRY_LINK([$3],[$8],[
eval $1_LIB_DIR='"$6"'

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
eval $1_LIB_DIR=''
eval $1_LIB_DIR_FLAG=''
eval $1_LIB_FLAG=''
eval HAVE_$1="no"
eval INC_$1_H=''
eval HAVE_$1_H="no"
AC_MSG_RESULT(not found)
if test "$9" = "not-optional"; then
  SCI_MSG_ERROR([[Test for required $1 failed. 
    To see the failed compile information, look in config.log, 
    search for $1. Please install the relevant libraries
     or specify the correct paths and try to configure again.]])
fi
])

#restore variables
CFLAGS=$_sci_savecflags
CXXFLAGS=$_sci_savecxxflags
LDFLAGS=$_sci_saveldflags
LIBS=$_sci_savelibs
_sci_lib_path=''
_sci_libs=''
_sci_includes=''

])


AC_DEFUN([SCI_TRY_LINK], [
##
## SCI_TRY_LINK ($1):  $2
##
## arguments mean:
## arg 1 : variable base name (e.g., MATH)
## arg 2 : checking message
## arg 3 : includes that arg 8 needs to compile (e.g., math.h)
##           If the first arg is "extern_C", then extern "C" is added around includes.
## arg 4 : include path(s). -I is appended to each path (unless it already has a -I).
##           Any other -? args are removed.  (-faltivec is ok on mac)
##           (Only one may be given if "Specific" (arg9) is chosen.)
## arg 5 : list of libs to link against
##           If the libraries do not have a '-l' on them, it is appeneded.
##           If the arg has any prefix besides '-l' (eg: -L or -D), then the arg
##           is removed completely.  (-framework is also allowed for Mac support.)
##           All the libs specified in arg 5 will be part of the <VAR>_LIB_FLAG
##           if the link is successful.
## arg 6 : lib paths
##           If the arg does not have -L on it, then -L will be added.  
##           If it has any other -?, then the arg is removed.
##           (only one may be givein if "Specific" (arg9) is chosen.)
## arg 7 : extra link flags 
##           This is where to put anything else you need for the compilation
##           line.  NOTE, none of these args propagate anywhere.
## arg 8 : body of code to compile. May be EMPTY, in which case a dummy routine is tried.
## arg 9 : 'optional' or 'required' or 'specific' required argument
##             If specific, then SCI_TRY_LINK will take only one lib
##             path and one include path and will verify that the
##             libs/includes are in that path.
##
## Here are the specific values for this invocation:
##
## arg 1 : $1
## arg 2 : $2
## arg 3 : $3
## arg 4 : $4
## arg 5 : $5
## arg 6 : $6
## arg 7 : $7
## arg 8 : <Can't expand it here as it may be multiple lines and cause sh problems.>
## arg 9 : $9
## 
## after execution of macro, the following will be defined:
##      Variable            Value
##      --------            -----
##      $1_LIB_DIR        => lib path
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
     echo
     AC_MSG_ERROR(Last parameter of SCI-TRY-LINK for $2 must be: optional or required or specific.  (You had $9.)  This is an internal SCIRun configure error and should be reported to scirun-develop@sci.utah.edu.)
fi

# If $4 (the -I paths) is blank, do nothing, else do the for statement.
ifelse([$4],[],,[
for inc in $4; do

  if test "$inc" = "/usr/include" || test "$inc" = "-I/usr/include"; then
     echo ""
     AC_MSG_ERROR(Please do not specify /usr/include as the location for $1 include files.)
  fi

  # Make sure it doesn't have any thing but -I
  #   The following "sed" replaces anything that starts with a '-' with nothing (blank). 
  has_minus_faltivec=no
  has_minus=`echo $inc | sed 's/-.*//'`
  if test -z "$has_minus"; then
     has_minus_i=`echo $inc | sed 's/-I.*//'`
     has_minus_faltivec=`echo $inc | sed 's/-faltivec//'`
     if test -n "$has_minus_i" && test -n "$has_minus_faltivec"; then
        # Has some other -?.
        if test "$debugging" = "yes"; then
          echo
          AC_MSG_WARN(Only -I options are allowed in arg 4 ($4) of $1 check.  Skipping $inc.)
        fi
        continue
     fi
  fi

  the_inc=`echo $inc | grep "^\-I"`
  if test -z "$the_inc" && test "$has_minus_faltivec" = "no"; then
     # If the include arg does not already have -I on it.
     if test -d $inc; then
        # If the directory exists
        _sci_includes="$_sci_includes -I$inc"
     fi
  else
     # It already has -I so just add it directly.
     _sci_includes="$_sci_includes $inc"
  fi
done
])dnl

### Take care of arg 5 (the list of libs)

if test -n "$5"; then

   found_framework=no
   for lib in $5; do

      if test "$found_framework" = "one"; then
         found_framework=two
      else
         found_framework=no
      fi

      # Make sure it doesn't have any thing but -l
      has_minus=`echo $lib | sed 's/-.*//'`
      if test -z "$has_minus"; then
         has_minus_l=`echo $lib | sed 's/-l.*//'`
         has_minus_framework=`echo $lib | sed 's/-framework.*//'`

         if test -z "$has_minus_framework"; then
            # Two rounds for this loop with respect to frameworks.
            # First round is to skip adding -l to the beginning of -framework.
            # Second round is to not add -l to the framework lib.
            found_framework=one
         fi

         if test -n "$has_minus_l" && test -n "$has_minus_framework"; then
            # Has some other -?.
            if test "$debugging" = "yes"; then
              echo
              AC_MSG_WARN(Only -l options are allowed in arg 5 of $1 check (disregarding $lib).)
            fi
            continue
         fi
      fi

      the_lib=`echo $lib | grep "^\-l"`
      if test -z "$the_lib" && test "$found_framework" = "no"; then
         # If the lib arg does not have -l on it, then add -l.
         if test "$IS_VC" = "yes"; then
           if test -n "`echo $lib | sed 's,.*\.lib,,'`"; then
             final_lib=$lib.lib
           else
             final_lib=$lib
           fi
         else
           final_lib=-l$lib
         fi
      else
         # It already has -l so just add it directly.
         final_lib=$lib
      fi
      _sci_libs="$_sci_libs $final_lib"
   done
fi
   
### Take care of arg 6 (the list of lib paths)

if test -n "$6"; then

   for path in $6; do

      # Make sure it doesn't have any thing but -L
      has_minus=`echo $path | sed 's/-.*//'`
      if test -z "$has_minus"; then
         has_minus_L=`echo $path | sed 's/-L.*//'`
         if test -n "$has_minus_L"; then
            # Has some other -?.
            if test "$debugging" = "yes"; then
              echo
              AC_MSG_WARN(Only -L options are allowed in arg 6 of $1 check (disregarding $path).)
            fi
            continue
         fi
      fi
   
      # Remove the '-L' (if it has one).
      the_path=`echo $path | sed 's/-L//'`
      if test -d "$the_path"; then
         _sci_lib_path="$_sci_lib_path $LDRUN_PREFIX$the_path -L$the_path"
      else
         echo
         AC_MSG_WARN(The path given $the_path is not a valid directory... ignoring.)
      fi
   done
fi

if test "$9" = "specific"; then
  # If 'specific' then only one lib is allowed:
  ### Determine if there is only one item in $6 (I don't know of a better way to do this.)
  __sci_pass=false
  __sci_first_time=true

  # Must have the "" for the SGI sh.
  for value in "" $6; do
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
       AC_MSG_ERROR(For specific SCI-TRY-LINK test for $1 only one library path may be specified for arg 6 (you had: $6).  This is an internal SCIRun configure error and should be reported to scirun-develop@sci.utah.edu.)
  fi
  # and only one include path:
  ### Determine if there is only one item in $4
  __sci_pass=false
  __sci_first_time=true
  for value in "" $4; do
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
  if test -n "$4" && test $__sci_pass != "true"; then
       AC_MSG_ERROR(For specific SCI-TRY-LINK test for $1 only one include path may be specified for arg 4 (you had: $4).  This is an internal SCIRun configure error and should be reported to scirun-develop@sci.utah.edu.)
  fi
fi

### Debug messages:
#echo "sci_includes: $_sci_includes"
#echo "sci_libs: $_sci_libs"
#echo "sci_lib_path: $_sci_lib_path"

CFLAGS="$_sci_includes $CFLAGS"
CXXFLAGS="$_sci_includes $CXXFLAGS"
if test "$IS_VC" = "yes"; then
  oldLIB=$LIB
  export LIB="`echo $_sci_lib_path | sed 's, -LIBPATH:,;,g' | sed 's,-LIBPATH:,,g' | sed 's, -L,;,g' | sed 's,-L,,g'`;$LIB"
else
  LDFLAGS="$_sci_lib_path $LDFLAGS"
fi

LIBS="$_sci_libs $7 $LIBS"

# Build up a list of the #include <file> lines for use in compilation:
__extern_c="no"
__sci_pound_includes=""

for inc in "" $3; do
    if test "$inc" = "extern_C"; then
       __sci_pound_includes="extern \"C\" {"
       __extern_c=yes
    else
      # Have to have the "" for the SGI sh. 
      if test "$inc" = ""; then
        continue
      fi
      __sci_pound_includes="$__sci_pound_includes
#include <$inc>"
    fi
done

if test "$__extern_c" = "yes"; then
    __sci_pound_includes="$__sci_pound_includes
}"
fi

AC_TRY_LINK($__sci_pound_includes,[$8],[

if test "$IS_VC" = "yes"; then
  LIB=$oldLIB
fi

eval $1_LIB_DIR='"$6"'

# Remove any bad (/usr/lib) lib paths and the thirdparty lib path
_final_dirs=
for _dir in $_sci_lib_path; do
  if test -n "$_dir" && test "$_dir" != "-L/usr/lib" ; then
    _final_dirs="$_final_dirs $_dir"
  fi
done

# Remove the thirdparty rpath stuff (if it exists) (and /usr/lib rpath)

_final_dirs=`echo "$_final_dirs" | sed "s%$LDRUN_PREFIX/usr/lib %%g"`

# Remove leading spaces
_final_dirs=`echo "$_final_dirs" | sed "s/^ *//"`

eval $1_LIB_DIR_FLAG="'$_final_dirs'"

# Remove any -L from the list of libs.  (-L's should only be in the dir path.)
final_libs=
for _lib in "" $_sci_libs; do
  bad_l_arg=`echo "$_lib" | grep "^\-L"`
  bad_i_arg=`echo "$_lib" | grep "^\-I"`
  if test -n "$_lib" && test "$_lib" != "/usr/lib" && test -z "$bad_l_arg" && test -z "$bad_i_arg"; then
    final_libs="$final_libs $_lib"
  fi
done

# Remove leading spaces
final_libs=`echo $final_libs | sed "s/^ *//"`
eval $1_LIB_FLAG="'$final_libs'"
eval HAVE_$1="yes"

final_incs=
for inc in "" $_sci_includes; do
   if test "$inc" != "$INC_SCI_THIRDPARTY_H"; then
      final_incs="$final_incs $inc"
   fi
done

# Remove leading spaces
final_incs=`echo $final_incs | sed "s/^ *//"`

eval INC_$1_H="'$final_incs'"
eval HAVE_$1_H="yes"

AC_MSG_RESULT(yes)
], 
[
eval $1_LIB_DIR=''
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
  for i in "" $3; do
    #echo looking for $4/$i
    if test ! -e $4/$i; then
     AC_MSG_ERROR(Specifically requested $1 include file '$4/$i' was not found)
    fi
  done
  # Make sure the exact libraries were found
  for i in "" $5; do
    if test -z "$i"; then
       continue
    fi
    has_minus=`echo $i | sed 's/-.*//'`
    if test -z "$has_minus"; then
       i=`echo $i | sed 's/-l//g'`
    fi
    if test ! -e $6/lib$i.so && test ! -e $6/lib$i.a && test ! -e $6/$i.lib && test ! -e $6/lib$i.dylib; then
     AC_MSG_ERROR(Specifically requested $1 library file '$6/lib$i' (.so, .a, or .dylib) was not found)
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

##
## END of SCI_TRY_LINK ($1):  $2
##

])


## SCI_CUDA_COMPILE_LINK_TEST
AC_DEFUN([SCI_CUDA_COMPILE_LINK_TEST], [
## arguments mean:
## arg 1  : variable base name i.e. CUDA
## arg 2  : checking message
## arg 3  : includes that arg 8 needs to compile
## arg 4  : include paths -I
## arg 5  : list of libs to link against -l
## arg 6  : lib paths -L
## arg 7  : extra link flags
## arg 8  : body of code (source file) to compile
## arg 9  : optional or not-optional required argument
## arg 10 : whether or not to set env vars
##
## after execution of macro, the following will be defined:
##      Variable            Value
##      --------            -----
##      $1_LIB_DIR        => lib path
##      $1_LIB_DIR_FLAG   => all the -L's
##      $1_LIB_FLAG       => all the -l's
##      HAVE_$1           => yes or no
##      INC_$1_H          => all the -I's
##      HAVE_$1_H         => yes or no

# make sure we have a base name, bail otherwise
ifelse([$1],[],[AC_FATAL(must provide a test name in arg 1)],)dnl

ifelse([$9],[optional],,[$9],[not-optional],,
       [AC_FATAL(arg 9 must be either 'optional' or 'not-optional' got $9)])dnl

ifelse([$10],[yes],,[$10],[no],,
       [AC_FATAL(arg 10 must be either 'yes' or 'no' got $10)])dnl       

AC_MSG_CHECKING(for $2 ($9))
echo

# save the precious variables
_sci_savecc=$CC
_sci_savelibs=$LIBS
_sci_saveldflags=$LDFLAGS
_sci_savecflags=$CFLAGS
_sci_savecxxflags=$CXXFLAGS

_sci_includes=
ifelse([$4],[],,[
for i in $4; do
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

# Look for the CUDA compiler, "nvcc"
AC_PATH_PROG([NVCC], [nvcc], [no], [$with_cuda/bin])

# Compile for Fermi GPUS:
#   The line below will compile the kernels specifically for Fermi architecture once and for all.
#   If the flag is not specified, then the kernels must be recompiled at runtime every time.
NVCC_CXXFLAGS="-gencode arch=compute_20,code=sm_20 "

# set up the -Xcompiler flag so that NVCC can pass CXXFLAGS to host C++ comiler
for i in $CXXFLAGS; do
  NVCC_CXXFLAGS="$NVCC_CXXFLAGS -Xcompiler $i"
done

NVCCEXT="ptx"

if test "$debug" = "yes"; then
  # The "-g -G0" option pair must be passed to NVCC when an application is compiled in
  # order to debug with cuda‐gdb. This forces -O0 compilation, with the exception of
  # very limited dead‐code eliminations and register‐spilling optimizations.
  NVCC_CXXFLAGS="-g -G0 $NVCC_CXXFLAGS $_sci_includes"
else
  NVCC_CXXFLAGS="$NVCC_CXXFLAGS $_sci_includes"
fi
NVCC_LIBS="$_sci_lib_path $_sci_libs"

# check that the CUDA compiler works
_file_base_name=`echo $8 | sed 's/\(.*\)\..*/\1/'`

AC_MSG_CHECKING([for C++ compilation using nvcc])
$NVCC $NVCC_CXXFLAGS -c $8
if test -f $_file_base_name.o; then
  AC_MSG_RESULT([yes])
else
	AC_MSG_RESULT([no])
	AC_MSG_ERROR( [For some reason we could not compile using nvcc] )
fi

# check we can also link via C++ compiler
AC_MSG_CHECKING([for linking nvcc compiled object code via C++ compiler])
$CXX $NVCC_LIBS -o $_file_base_name $_file_base_name.o

if test -f $_file_base_name; then
  AC_MSG_RESULT([yes])
else
  AC_MSG_RESULT([no])
  AC_MSG_ERROR( [For some reason we could not link nvcc compiled object code] )
fi

AC_MSG_CHECKING([for NVCC compilation to PTX])
$NVCC -ptx -gencode arch=compute_20,code=sm_20 $8
if test -f $_file_base_name.$NVCCEXT; then
  AC_MSG_RESULT([yes])
  HAVE_CUDA="yes"
else
  AC_MSG_RESULT([no])
  AC_MSG_ERROR( [For some reason we could not compile to PTX using nvcc] )
  HAVE_CUDA="no"
fi

if test $HAVE_CUDA="yes"; then
  DEF_CUDA="#define HAVE_CUDA 1"
  eval $1_LIB_DIR='"$6"'
  eval $1_LIB_DIR_FLAG='"$_sci_lib_path"'
  eval $1_LIB_FLAG='"$_sci_libs"'

  if test "$_sci_includes" = "$INC_SCI_THIRDPARTY_H"; then
    eval INC_$1_H=''
  else
    eval INC_$1_H='"$_sci_includes"'
  fi
  eval HAVE_$1_H="yes"

else
  eval $1_LIB_DIR=''
  eval $1_LIB_DIR_FLAG=''
  eval $1_LIB_FLAG=''
  eval HAVE_$1="no"
  eval INC_$1_H=''
  eval HAVE_$1_H="no"
fi

if test ! "$DEF_CUDA"; then
   echo
   SCI_MSG_ERROR(one or more of the CUDA components is missing.)
fi

if test "$9" = "not-optional"; then
  SCI_MSG_ERROR([[Test for required $1 failed.
    To see the failed compile information, look in config.log,
    search for $1. Please install the relevant libraries
     or specify the correct paths and try to configure again.]])
fi

#restore previous precious variables
CC=$_sci_savecc
CFLAGS=$_sci_savecflags
CXXFLAGS=$_sci_savecxxflags
LDFLAGS=$_sci_saveldflags
LIBS=$_sci_savelibs
_sci_includes=''
_sci_lib_path=''
_sci_libs=''

##
## END of SCI_COMPILE_LINK_CUDA_TEST ($1):  $2
##

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
      if test "$_CUR_2_" $_SCI_COMP_ "$_CUR_1_"; then
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
##  SCI_REMOVE_MINUS_L(lib_flag,dirs,libs)
##
##  lib_flag     : variable to set with the final list of libs.
##  dirs         : List of dirs (eg: -L/usr/X11R6/lib -L/usr/lib -L/sci/thirdparty)
##  libs         : List of libs (eg: -lm -lX11)
##
##  Parses a library include line ('dirs') looking for each library (in
##  'libs').  Assigns to the variabl 'lib_flag' the exact library file
##  (instead of using -l) (if it is found in one of the 'dirs' directories.
##  For example, with the example lines given above, the final output would be:
## 
##  lib_flag:      -lm /usr/X11R6/lib/libX11.so
##
##  If a '.so' and a '.a' file exist in the lib directory, the .so is given 
##  preference.  (We also check for .dylib files.)

AC_DEFUN(SCI_REMOVE_MINUS_L,
  [
    ##  SCI_REMOVE_MINUS_L

   dirs="$2"
   libs="$3"

   if test -z "$dirs" || test -z "$libs"; then
       AC_MSG_ERROR(The dirs '$dirs' and/or libs '$libs' parameters of SCI-REMOVE-MINUS-L are empty.  This is an internal SCIRun configure error and should be reported to scirun-develop@sci.utah.edu.)
   else

     sci_temp_lib=

     # list of libraries that have been found
     found_libs=

     got_it=
     for libflag in $libs; do

       # If the entry starts with -L, then we ignore it... (All libs should be -l!)
       has_minus_L=`echo $libflag | sed 's/-L.*//'`
       if test -n "$has_minus_L"; then
          AC_MSG_WARN(Only -L options are allowed in arg 3 (disregarding $libflag).)
          continue
       fi

       # Cut of the '-l'
       the_lib=lib`echo $libflag | sed 's/-l//'`
  
       checked_dirs=
       for dirflag in $dirs; do

         # Cut of the '-L' from each library directory.
         the_dir=`echo $dirflag | sed 's/-L//'`
  
         # Disregard generic libraries:
         if test "$the_dir" = "/usr/lib"; then
           continue
         fi

         already_checked=`echo \"$checked_dirs\" | grep "$the_dir "`
         if test -n "$already_checked"; then
           AC_MSG_WARN($the_dir listed more than once as a -L flag.  Skipping.)
           continue
         else
           checked_dirs="$checked_dirs$the_dir "
         fi

         sci_found=
         # Check to see if the .so exists in the given directory
         if test -e $the_dir/$the_lib.so; then
           sci_temp_lib="$sci_temp_lib $the_dir/$the_lib.so"
           sci_found=$the_lib
         else
           # If no .so, then look for a '.dylib' file.
           if test -e $the_dir/$the_lib.dylib; then
              sci_temp_lib="$sci_temp_lib $the_dir/$the_lib.dylib"
              sci_found=$the_lib
           else
             # If no .so, then look for a '.a' file.
             if test -e $the_dir/$the_lib.a; then
                sci_temp_lib="$sci_temp_lib $the_dir/$the_lib.a"
                sci_found=$the_lib
             fi
           fi
         fi

         if test -n "$sci_found"; then
           already_found=`echo $found_libs | grep "$sci_found "`
         fi

         if test -n "$already_found"; then
           AC_MSG_ERROR($libflag found in more than one location -- $the_dir and $already_found!)
         else
           if test -n "$sci_found"; then
             found_libs="$found_libs$the_dir/$sci_found "
             got_it=true
           fi
         fi
       done

       if test -z "$got_it"; then
         # Add -l<lib> flag to line as it is a generic lib.
         sci_temp_lib="$sci_temp_lib $libflag"
       fi
       got_it=
     done 
     $1=$sci_temp_lib

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
      if test "$_CUR_2_" $_SCI_COMP_ "$_CUR_1_"; then
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
## calls AC_ARG_VAR, makes variables precious, and allows the vars to pass
## our valid check by recording it in sci_arg_var_list
##
AC_DEFUN([SCI_ARG_VAR], [
  AC_ARG_VAR($1, $2)
  sci_arg_var_list="$sci_arg_var_list $1"
])

