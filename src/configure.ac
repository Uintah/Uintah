##
##  The contents of this file are subject to the University of Utah Public
##  License (the "License"); you may not use this file except in compliance
##  with the License.
##  
##  Software distributed under the License is distributed on an "AS IS"
##  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
##  License for the specific language governing rights and limitations under
##  the License.
##  
##  The Original Source Code is SCIRun, released March 12, 2001.
##  
##  The Original Source Code was developed by the University of Utah.
##  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
##  University of Utah. All Rights Reserved.
##

##  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##
##  This configure.in file has some specialized sections for doing certain
##  tasks.  If you want to add additional tasks (search for new libraries,
##  headers, etc.) please add the new task to the appropriate section.
##  Please also try to adhere to the conventions used in this file
##  for standard functionality and readability.
##
##  The following is a list of the current sections:
##
##    initialize
##    new macro definitions
##    declare arguments  (the --with and --enable stuff)
##    check for unknown or mis-spelled arguments
##    determine host type, processor count and OS version
##    set variable defaults 
##    set host specific build tools
##    search for required libraries and headers
##    search for optional libraries and headers
##    configure packages
##    generate output files
##
##  NOTE: When committing this file, do NOT commit both configure.in
##  and configure at the same time.  Please commit configure.in FIRST,
##  then commit configure separately.  It saves people from conflicts
##  with autoconf.
##   - Steve
##
##  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##  --------------------------------------------------------------------
##  ------------------------  initialize  ------------------------------
##  --------------------------------------------------------------------

##  initialize configure 
AC_INIT([SCIRun],[1.10],[scirun-users@sci.utah.edu])

##  require autoconf version 2.53
AC_PREREQ(2.53)

AC_COPYRIGHT(
##
##  The contents of this file are subject to the University of Utah Public
##  License (the "License"); you may not use this file except in compliance
##  with the License.
##  
##  Software distributed under the License is distributed on an "AS IS"
##  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
##  License for the specific language governing rights and limitations under
##  the License.
##  
##  The Original Source Code is SCIRun, released March 12, 2001.
##  
##  The Original Source Code was developed by the University of Utah.
##  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
##  University of Utah. All Rights Reserved.
##
)

## Clean up files that could be left around if configure was 
## interrupted (by control-c, ^c) because I do not know how to add
## cleanup to the builtin AC function that uses the 'trap' command.
rm -rf ii_files sedscript

AC_CONFIG_SRCDIR(configVars.mk.in)

##  set the location of config.guess, config.sub, and install.sh
AC_CONFIG_AUX_DIR(scripts)

##  set the output header to "sci_config.h"
AC_CONFIG_HEADER(sci_config.h)

## Cut off the /configure from the configure command (eg: ../src/configure)
path_to_src=`echo $0 | sed 's/\/configure//'`

## If user tried to configure in current directory, this will block it.
if test "$path_to_src" = "configure" -o "$path_to_src" = "."; then
   AC_MSG_WARN(DO NOT CONFIGURE IN THE SRC DIRECTORY:)
   AC_MSG_WARN(___ Try creating a dir such as: ../archBITSdebug)
   AC_MSG_WARN(___ eg: ../sgi32opt or ../linux32debug)
   AC_MSG_WARN(___ cd to that dir and then type: ../src/configure)
   rm -f config.log config.cache confdefs.h conftest.out
   AC_MSG_ERROR(Tried to configure in src directory.)
fi

# make sure these variables are considered precious.
SCI_ARG_VAR(CC, C compiler command)
SCI_ARG_VAR(CFLAGS, C compiler flags)
SCI_ARG_VAR(LDFLAGS, linker flags, e.g. -L<lib dir> if you have libraries in a\
            nonstandard directory <lib dir>)
SCI_ARG_VAR(CXX, C++ compiler command)
SCI_ARG_VAR(CPPFLAGS, C/C++ preprocessor flags, e.g. -I<include dir> if you \
            have headers in a nonstandard directory <include dir>)
SCI_ARG_VAR(CXXFLAGS,C++ compiler flags)
SCI_ARG_VAR(F77,Fortran 77 compiler command)
SCI_ARG_VAR(FFLAGS,Fortran 77 compiler flags)
SCI_ARG_VAR(CPP,C preprocessor)
SCI_ARG_VAR(CXXCPP,C++ preprocessor)

# Initialize variables that control what optional checks to make
## all the optional tests we have... These get set in the macro 
# SCI_SET_PACKAGE_CHECKS found in aclocal.m4.  Each package sets the 
# checks that they 
INIT_PACKAGE_CHECK_VARS()


## find the machine's pwd, and don't use the sh built-in pwd.
## sh pwd can print the wd with symbolic links depending on
## how you got to that directory, and thus has the potential
## to print different pwd's for the same wd, and change files
## like sci_defs.h

sci_pwd=""
for my_pwd in pwd /bin/pwd /usr/bin/pwd; do
  mkdir pwd_test
  ln -s pwd_test pwd_temp

  cd pwd_test
  pwd_1=`$my_pwd`

  cd ../pwd_temp
  pwd_2=`$my_pwd`

  cd ..
  rm -r pwd_test pwd_temp
  if test "$pwd_1" = "$pwd_2"; then
    sci_pwd=$my_pwd
    alias pwd="$my_pwd"
    break
  fi
done
if test -z "$sci_pwd"; then
   AC_MSG_WARN("Using unstable version of pwd")
   sci_pwd=pwd
fi


##  initialize the recognized argument lists
sci_arg_with_list="--no-create --no-recursion"
sci_arg_enable_list=""

##  --------------------------------------------------------------------
##  ------------------  check for all arguments     --------------------
##  --------------------------------------------------------------------

##
##  Use the SCI_ARG_WITH and SCI_ARG_ENABLE macros to check for arguments
##

SCI_ARG_WITH(thirdparty, 
  AC_HELP_STRING(--with-thirdparty=DIR,
                Use the Thirdparty_install installation in DIR. This is output from the Thirdparty_install install script. The default is /usr/local/SCIRun/Thirdparty/ followed by: <version>/<OS>/<compiler>-<32 or 64>bit),
  [thirdparty="$withval"],
  [thirdparty=""])
SCI_ARG_WITH(blas,
  AC_HELP_STRING(--with-blas=DIR,
                 Use the blas installation in DIR),
  [with_blas="$withval"
   sci_check_blas=yes],
  [with_blas=""])
SCI_ARG_WITH(lapack,
  AC_HELP_STRING(--with-lapack=DIR, 
                 Use the lapack installation in DIR),
  [with_lapack="$withval"
   sci_check_lapack=yes],
  [with_lapack=""])
SCI_ARG_WITH(mpi,
  AC_HELP_STRING(--with-mpi=DIR,
                 Use the mpi installation in DIR),
  [with_mpi="$withval"
   sci_check_mpi=yes],
  [with_mpi=""])
SCI_ARG_WITH(mpiuni,
  AC_HELP_STRING(--with-mpiuni=DIR,
                 Use the mpiuni installation in DIR),
  [with_mpiuni="$withval"],
  [with_mpiuni=""])
SCI_ARG_WITH(hypre,
  AC_HELP_STRING(--with-hypre=DIR,
                 Use the hypre installation in DIR),
  [with_hypre="$withval"
   sci_check_hypre=yes],
  [with_hypre=""])
SCI_ARG_WITH(netsolve,
  AC_HELP_STRING(--with-netsolve=DIR,
                 Use the netsolve installation in DIR),
  [with_netsolve="$withval"
   sci_check_netsolve=yes],
  [with_netsolve=""])
SCI_ARG_WITH(petsc,
  AC_HELP_STRING(--with-petsc=DIR,
                 Use the petsc installation in DIR),
  [with_petsc="$withval"
   sci_check_petsc=yes],
  [with_petsc=""])
SCI_ARG_WITH(unipetsc,
  AC_HELP_STRING(--with-unipetsc=DIR,
                 Use the unipetsc installation in DIR),
  [with_unipetsc="$withval"
   sci_check_unipetsc=yes],
  [with_unipetsc=""])
SCI_ARG_WITH(tau,
  AC_HELP_STRING(--with-tau=<tau-makefile>,
                 Specify a Tau Makefile),
  [with_tau="$withval"
   sci_check_tau=yes],
  [with_tau=""])
SCI_ARG_WITH(globus,
  AC_HELP_STRING(--with-globus=DIR,
                 Use the globus installation in DIR),
  [with_globus="$withval"
   sci_check_globus=yes
   sci_check_ssl=yes
   sci_check_crypto=yes],
  [with_globus=""])
SCI_ARG_WITH(crypto,
  AC_HELP_STRING(--with-crypto=DIR,
                 Use the crypto installation in DIR (only for use with globus)),
  [with_crypto="$withval"
   sci_check_crypto=yes],
  [with_crypto=""])
SCI_ARG_WITH(ssl,
  AC_HELP_STRING(--with-ssl=DIR,
                 Use the ssl installation in DIR (only for use with globus)),
  [with_ssl="$withval"
   sci_check_ssl=yes],
  [with_ssl=""])
SCI_ARG_WITH(insight,
  AC_HELP_STRING(--with-insight=DIR,
	         Use the Insight installation in DIR),
  [with_insight="$withval"
   sci_check_insight=yes],
  [with_insight=""])
SCI_ARG_WITH(qt,
  AC_HELP_STRING(--with-qt=DIR,
                 Use the Qt installation in DIR),
  [with_qt="$withval"
   sci_check_qt=yes],
  [with_qt=""])
SCI_ARG_WITH(opengl,
  AC_HELP_STRING(--with-opengl=DIR,
                 Use the OpenGL installation in DIR),
  [with_opengl="$withval"],
  [with_opengl=""])
SCI_ARG_WITH(zlib,
  AC_HELP_STRING(--with-zlib=DIR,
                 Use the zlib installation in DIR),
  [with_zlib="$withval"],
  [with_zlib=""])
SCI_ARG_WITH(jpeg,
  AC_HELP_STRING(--with-jpeg=DIR,
                 Use the jpeg installation in DIR),
  [with_jpeg="$withval"
   sci_check_jpeg=yes],
  [with_jpeg=""])
SCI_ARG_WITH(tiff,
  AC_HELP_STRING(--with-tiff=DIR,
                 Use the tiff installation in DIR),
  [with_tiff="$withval"
   sci_check_tiff=yes],
  [with_tiff=""])

SCI_ARG_WITH(xalan,
  AC_HELP_STRING(--with-xalan=DIR,
                 Use the xalan installation in DIR),
  [with_xalan="$withval"
   sci_check_xalan=yes],
  [with_xalan=""])

SCI_ARG_WITH(mdsplus,
  AC_HELP_STRING(--with-mdsplus=DIR,
                 Use the mdsplus installation in DIR),
  [with_mdsplus="$withval"
   sci_check_mdsplus=yes],
  [with_mdsplus=""])

SCI_ARG_WITH(hdf5,
  AC_HELP_STRING(--with-hdf5=DIR,
                 Use the hdf5 installation in DIR),
  [with_hdf5="$withval"
   sci_check_hdf5=yes],
  [with_hdf5=""])

SCI_ARG_WITH(glut,
  AC_HELP_STRING(--with-glut=DIR,
                 Use the glut installation in DIR),
  [with_glut="$withval"
   sci_check_glut=yes],
  [with_glut=""])
SCI_ARG_WITH(glui,
  AC_HELP_STRING(--with-glui=DIR,
                 Use the glui installation in DIR),
  [with_glui="$withval"
   sci_check_glui=yes],
  [with_glui=""])
SCI_ARG_WITH(oogl,
  AC_HELP_STRING(--with-oogl=DIR,
                 Use the oogl installation in DIR),
  [with_oogl="$withval"
   sci_check_oogl=yes],
  [with_oogl=""])
SCI_ARG_WITH(audio,
  AC_HELP_STRING(--with-audio=DIR,
                 Use the audio installation in DIR),
  [with_audio="$withval"
   sci_check_audio=yes],
  [with_audio=""])

SCI_ARG_ENABLE(verbosity,
  AC_HELP_STRING(--enable-verbosity,
                 Turn on verbose configure output),
  [debugging="yes"],
  [debugging="no"])
SCI_ARG_ENABLE(threads,
  AC_HELP_STRING(--enable-threads,
                 Specify a thread implementation (pthreads or irix)),
  [threads="$enableval"],
  [threads=${threads='unknown'}])
SCI_ARG_ENABLE(debug,
  AC_HELP_STRING(--enable-debug,
                 Turn on debugging (default is -g) or specify your own.
                 example: --enable-debug='-O0 -g'),
  [debug="$enableval"],
  [debug="no"])
SCI_ARG_ENABLE(optimize,
  AC_HELP_STRING(--enable-optimize,
                 Turn on optimize (default is -O2) or specify your own.
                 example: --enable-optimize='-O3'),
  [optimize="$enableval"],
  [optimize=${optimize='no'}])
SCI_ARG_ENABLE(64bit, AC_HELP_STRING(--enable-64bit,
                 Compile in 64 bit mode),
  [enable_64bit="$enableval"],
  [enable_64bit=$i{enable_64bit='no'}])
SCI_ARG_ENABLE(32bit,
  AC_HELP_STRING(--enable-32bit,
                 Compile in 32 bit mode),
  [enable_32bit="$enableval"],
  [enable_32bit=$i{enable_32bit='no'}])
SCI_ARG_ENABLE(output-avg-walltime,
  AC_HELP_STRING(--enable-output-avg-walltime,
                 Output ellapsed wall-times),
  [output_avg_walltime="$enableval"],
  [output_avg_walltime=${output_avg_walltime='no'}])
SCI_ARG_ENABLE(perfex,
  AC_HELP_STRING(--enable-perfex,
                 Use perfex hardware counters (sgi only)),
  [sci_check_perfex=yes],
  [sci_check_perfex=no])
SCI_ARG_ENABLE(package,
  AC_HELP_STRING(--enable-package=<list>,
                 Use the SCIRun packages specified by list.
                 example: --enable-package='BioPSE Teem'
                 default: all directories under SCIRun/src/Packages),
  [package="$enableval"],
  [
   AC_MSG_WARN("Enabling all the packages in your SCIRun/src/Packages directory.")
   _sci_tmp=`ls $path_to_src/Packages`
   package=`echo $_sci_tmp | sed 's/CVS //g'`
  ])
SCI_ARG_ENABLE(exename,
  AC_HELP_STRING(--enable-exename=<name>,
                 Use name for executable name instead of the default: scirun),
  [exename="$enableval"],
  [exename=${exename='default'}])
SCI_ARG_ENABLE(largesos,
  AC_HELP_STRING(--enable-largesos,
                 Build one .so per directory),
  [largesos="$enableval"],
  [largesos=${largesos='no'}])
SCI_ARG_ENABLE(sci-malloc,
  AC_HELP_STRING(--enable-sci-malloc,
                 Use SCI memory allocation/deallocation checks),
  [enable_sci_malloc="$enableval"],
  [enable_sci_malloc=${enable_sci_malloc='yes'}])
SCI_ARG_ENABLE(assertion-level,
  AC_HELP_STRING(--enable-assertion-level=N,
                 Use assertion level N (0-3)),
  [enable_assertion_level="$enableval"],
  [enable_assertion_level=${enable_assertion_level=3}])
SCI_ARG_ENABLE(scirun2,
  AC_HELP_STRING(--enable-scirun2,
                 Build experimental SCIRun 2),
  [enable_scirun2="$enableval"
   [SCI_SET_PACKAGE_CHECKS(SCIRun2)]
  ],
  [enable_scirun2=${enable_scirun2='no'}])


##  --------------------------------------------------------------------
##  Save off the arguments to configure, being careful to save 
##  Package strings for later parsing.
##  --------------------------------------------------------------------

args=`echo "$*" | sed 's% %_%g'`
myargs=`echo $args | sed 's%\^% %g'`
# Setting IFS breaks standard autoconf tests.  
# It thinks we are cross compiling. Don't set it.
#IFS=' '

PACKAGE_DIRS=""
# make sure the package dir exists
for i in $package; do
   if test ! -d "$path_to_src/Packages/$i"; then
      AC_MSG_ERROR(Package '$i' does not exist (in $path_to_src"Packages")!)
   fi
   PACKAGE_DIRS=$PACKAGE_DIRS" Packages/$i"
   # set the package dependent checks
   SCI_SET_PACKAGE_CHECKS($i)
done

if test "$sci_check_unipetsc" = "yes"; then
   sci_check_blas=yes
   sci_check_lapack=yes
fi

if test "$sci_check_blas" = "yes"; then
   sci_check_fortran=yes
fi


# Set the subdir name in the bin tree for matlab engine.
if test "$pkg_MatlabInterface" = "yes"; then
  MATLAB_ENG=matlab
  AC_SUBST(MATLAB_ENG)
fi

# Take care of package dependencies
if test "$pkg_Fusion" = "yes"; then
  # Fusion depends upon Teem Package.  
  if test -z $pkg_Teem; then
    AC_MSG_ERROR([The Fusion Package depends on the Teem Package. Please 
                  make sure you have the Teem Package and configure 
                  with it enabled.])
  fi
fi

LOAD_PACKAGE="SCIRun"
for i in $package; do
 LOAD_PACKAGE=$LOAD_PACKAGE,"$i"
done

DIRNAME_IS=`type dirname | sed 's%^.* %%'`
BASENAME_IS=`type dirname | sed 's%^.* %%'`

# link this way.
LDRUN_PREFIX="-Wl,-rpath -Wl,"

## We use both of these in a number of spots, so I think this is fair:
## (Eventually we could write our own if necessary...
if test -z "$DIRNAME_IS" -o -z "$BASENAME_IS"; then
  AC_MSG_ERROR(dirname or basename not found.  Configure script won't work!)
fi

##  --------------------------------------------------------------------
##  -----  determine host type, processor count and OS version  --------
##  --------------------------------------------------------------------

echo 
AC_CANONICAL_HOST
IS_OSX="no"
unset NUM_CPUS
case $host in
  *-irix*)
    SCI_CHECK_OS_VERSION(6.5,,
      configure: error: wrong Irix version. exiting.;exit)
    NUM_CPUS="`TERM=dumb top -n 1 | fgrep CPUs | cut -d' ' -f1`"    
    ;;
  *linux*)
    SCI_CHECK_OS_VERSION(2.2.17,,
      configure: error: wrong Linux version. exiting.;exit)
    NUM_CPUS="`cat /proc/cpuinfo | fgrep processor | wc | cut -c1-7`"
    ;;
  *-dec*)
    NUM_CPUS=4
    ;;
  *-darwin*)
    NUM_CPUS=2
    IS_OSX="yes"
    ;;
  *-aix*)
    NUM_CPUS=4
    ;;
  *)
    echo
    AC_MSG_WARN( !!!!!! This system is untested - You are on your own !!!!!!)
    echo
    NUM_CPUS="1"
    ;;
esac

AC_SUBST(IS_OSX)

AC_MSG_CHECKING(physical processor count)
if test -z "$NUM_CPUS"; then
  NUM_CPUS="1"
fi
AC_MSG_RESULT($NUM_CPUS)


if test "$CC" = ""; then
  # Apparently AC_PROG_CC will set CFLAGS to "-O2 -g" if you don't specify
  # its value first, so...
  CFLAGS=""
  AC_PROG_CC()
  AC_MSG_WARN(C compiler not specified with 'CC' environment variable.\
	    Using default... $CC)
fi

if test "$CXX" = ""; then
  case "$host" in 
     *-darwin*)
	CXX=c++
	;;
     *)
	# Apparently AC_PROG_CXX will set CXXFLAGS to "-O2 -g" if you don't specify
        # its value first, so...
        CXXFLAGS=""
        AC_PROG_CXX(CC g++ c++)
        AC_MSG_WARN(C++ compiler not specified with 'CXX' environment variable.\
	         Using default... $CXX)
        ;;
  esac
fi


if test $sci_check_fortran = "yes"; then

  if test "$F77" = ""; then
    AC_PROG_F77()
    AC_MSG_WARN(Fortran compiler not specified with 'F77' environment \
	    variable. Using default... $F77)
  fi
  # Most fortran compilers need an "_" for C function names.
  # Some have strange underscore needs.  This flag is used in the
  # Packages/Uintah/tools/fspec.pl.in file to generate the correct
  # cname of the fortran function.
  FORTRAN_UNDERSCORE=end
fi

CC_REALLY_IS=`type $CC | sed 's%^.* %%'`
# Going to look for "as", "ld", etc in this dir:
COMPILER_DIR=`dirname $CC_REALLY_IS`

## MAKE_PARALLELISM is used to tell the compiler how many cpus to use
## when linking.  1 is usually a good number to use per link,
## espcially for smaller number of cpu machines.  I think 4 is
## sufficient for even nirvana.
if test $NUM_CPUS -le 2 ; then
  MAKE_PARALLELISM=1
else
  if test $NUM_CPUS -le 8 ; then
    MAKE_PARALLELISM=2
  else
    MAKE_PARALLELISM=8
  fi
fi

AC_MSG_CHECKING(host system word length)
if test "$enable_64bit" = "yes"; then
  AC_MSG_RESULT(64 bits)
  if test "$enable_32bit" = "yes"; then
    SCI_MSG_ERROR(Both 64 and 32 bit compiles specified.)
  fi
else
  AC_MSG_RESULT(32 bits)
fi

##  --------------------------------------------------------------------
##  ----------------------  set variable defaults  ---------------------
##  --------------------------------------------------------------------

##
##  DO NOT "hardcode" ANY libraries (i.e. no -l or -L or -I allowed).
##

TIME_IMPL=Time_unix.cc
TEMPLATE_TAG=
TEMPLATE_BOX="<>"
case $host in
     *-darwin*)
	SO_OR_ARCHIVE_EXTENSION=dylib
	if test "$enable_sci_malloc" = "yes"; then
	   AC_MSG_WARN(sci_malloc does not work under mac osx. use -enable-sci-malloc=force)
	   AC_MSG_WARN(sci-malloc disabled)
	   enable_sci_malloc="no"
        elif test "$enable_sci_malloc" = "force"; then
	   enable_sci_malloc="yes"
        fi
	;;
	
     *-aix*)
        # On AIX, don't automatically generate .so's.  Make .a's and then
        # later we will make .so's from them.
	SO_OR_ARCHIVE_EXTENSION=a
	;;
     *)
	SO_OR_ARCHIVE_EXTENSION=so 
        ;;
esac

ASSERTION_LEVEL="$enable_assertion_level"

extra_cflags="";

if test "$output_avg_walltime" != "no"; then
  extra_cflags="-DOUTPUT_AVG_ELAPSED_WALLTIME ${extra_cflags}"
fi

if test "$exename" = "default"; then
  EXENAME=scirun
else
  EXENAME=$exename
fi

LARGESOS=$largesos

if test "$enable_sci_malloc" = "yes"; then
  DEF_MALLOC=""
else
  DEF_MALLOC="#define DISABLE_SCI_MALLOC"
fi

AC_SUBST(SO_OR_ARCHIVE_EXTENSION)
AC_SUBST(ASSERTION_LEVEL)
AC_SUBST(NOPRINTDIR)
AC_SUBST(DEF_MALLOC)
AC_SUBST(PACKAGE_DIRS)
AC_SUBST(LOAD_PACKAGE)
AC_SUBST(EXENAME)
AC_SUBST(LARGESOS)
AC_SUBST(TIME_IMPL)

#########################################################################

##  ----------------------------------------------------------------------
##  ----------------  set host specific build tools  ---------------------
##  ----------------------------------------------------------------------

##  DO NOT "hardcode" ANY libraries here.

##  Require GCC for all platforms except Irix which requires MIPSpro
##  and AIX which uses xlC

echo
AS_MESSAGE(checking for required build tools......)
echo 

AC_MSG_CHECKING(whether debug was enabled)
if test "$debug" = "no"; then
  AC_MSG_RESULT(no)
elif test "$debug" = "yes"; then
  extra_cflags="-g ${extra_cflags}"
  AC_MSG_RESULT(yes (-g) )
else
  extra_cflags="$debug ${extra_cflags}"
  AC_MSG_RESULT(yes ($debug) )
fi

AC_MSG_CHECKING(whether optimize was enabled)
if test "$optimize" = "no"; then
  if test "$debug" = "no"; then
    # neither debug nor optimize specified, DEFAULT TO: optimized
    extra_cflags="-O2 ${extra_cflags}"
    AC_MSG_RESULT(no (default to -O2) )
  else
    # Debug mode specified
    extra_cflags="${no_optimize_flag} ${extra_cflags}"
    # extra_fflags WILL ONLY BE USED WITH g77
    extra_fflags="-ffortran-bounds-check"
    AC_MSG_RESULT(no (${no_optimize_flag}) )
  fi
elif test "$optimize" = "yes"; then
  extra_cflags="-O2 ${extra_cflags}"
  AC_MSG_RESULT(yes (-O2) )
else
  extra_cflags="$optimize ${extra_cflags}"
  AC_MSG_RESULT(yes ($optimize) )
fi

if test "$sci_check_perfex" = "yes"; then
  extra_cflags="-DUSE_PERFEX_COUNTERS ${extra_cflags}"
fi

if test ! -e "`type $CC | sed 's%^.* %%'`"; then
  SCI_MSG_ERROR(Could not find the specified C compiler ($CC))
fi
if test ! -e "`type $CXX | sed 's%^.* %%'`"; then
  SCI_MSG_ERROR(Could not find the specified C++ compiler (CC))
fi

if test $sci_check_fortran = "yes"; then
  if test ! -e "`type $F77 | sed 's%^.* %%'`"; then
    SCI_MSG_ERROR(Could not find the specified Fortran compiler ($F77))
  fi
fi

case $host in 
  *-irix*)
    AC_MSG_WARN(Configure script only allows SGI compilers on IRIX.)
    AC_MSG_WARN(If this is a problem email scirun-users@sci.utah.edu.)

    ac_objext=o
    SCI_CHECK_VERSION($CC,-version,7.3.1.1,,
                      echo configure: error: wrong $CC version.; exit)
    SCI_CHECK_VERSION($CXX,-version,7.3.1.1,,
                      echo configure: error: wrong $CXX version.; exit)
    if test $sci_check_fortran = "yes"; then
      SCI_CHECK_VERSION($F77,-version,7.3.1.1,,
                        echo configure: error: wrong $F77 version.; exit)
    fi
    AC_PATH_PROG(AS,as,,$COMPILER_DIR,)
    if test -z "$AS"; then
      AC_MSG_WARN(Could not find the MIPSPro linker (as))
      AC_MSG_ERROR(Looked in $COMPILER_DIR)
    fi
    SCI_CHECK_VERSION($AS,-version,7.3.1.1,,
                      echo configure: error: wrong $CC version.; exit)
    AC_PATH_PROG(LD,ld,,$COMPILER_DIR,)
    if test -z "$LD"; then
      AC_MSG_WARN(Could not find the MIPSPro linker (ld))
      AC_MSG_ERROR(Looked in $COMPILER_DIR)
    fi
    CPP="$CC"
    CXXCPP="$CXX -c"
    ##  this WOFF is required for configure 
    WOFF=" -woff 3303,1047,1209,1110 -Wl,-woff,84 -woff 1342"
    CPPFLAGS=" $WOFF -LANG:std"
    if test "$enable_64bit" = "yes"; then
      BASE_LIB_DIR=lib64
      BINARYFLAGS="-64 -mips4"
    else
      BASE_LIB_DIR=lib
      BINARYFLAGS="-n32 -mips4"
    fi
    FFLAGS="$BINARYFLAGS $extra_cflags"
    CFLAGS="-fullwarn -xansi -LANG:std $BINARYFLAGS $extra_cflags"
    CXXFLAGS="$CFLAGS"
    ASFLAGS="$BINARYFLAGS"
    SOFLAGS="-shared $BINARYFLAGS -LANG:std -no_unresolved -J\$(MAKE_PARALLELISM) -update_registry \$(LIBDIR)so_locations $(TLINK) ${SOFLAGS}"
    LDFLAGS="$BINARYFLAGS -LANG:std -no_unresolved"
    AC_MSG_CHECKING(if ${CC} talks too much)
    touch wow.c
    if ${CC} -c wow.c 2>&1 | fgrep "requires a license password" > /dev/null ; then
      AC_MSG_RESULT(yes)
      AC_MSG_WARN(Using cc and CC wrappers in scripts/cc)

      mkdir -p scripts
      sed "s+VERBOSE_CC+${CC}+g;" < ${srcdir}/scripts/cc.in > scripts/cc
      sed "s+VERBOSE_CC+${CXX}+g;" < ${srcdir}/scripts/cc.in > scripts/CC
      chmod 755 scripts/cc scripts/CC

      # use the script to parse out license noise.  Must be absolute path so
      # dynamic compilation works.
      scripts_path=`cd scripts;pwd`
      CC="$scripts_path/cc"
      CXX="$scripts_path/CC"

    else
      AC_MSG_RESULT(no)
    fi
    rm -f wow.c wow.o
    ;;

  *-dec*)
    ac_objext=o

    CPP="$CC"
    CXXCPP="$CXX -c"
    ##  this WOFF is required for configure 
    CPPFLAGS=""
    FFLAGS="$extra_cflags"
    CFLAGS="-pthread $extra_cflags -D__USE_STD_IOSTREAM"
    CXXFLAGS="-model ansi -pthread $extra_cflags -D__USE_STD_IOSTREAM"
    SOFLAGS="-shared -warning_unresolved -update_registry \$(LIBDIR)so_locations"
    LDFLAGS=""
    LDRUN_PREFIX="-rpath "
    REPOSITORY_FLAGS="-ptr"
    ;;

  *-aix*)
  
    ac_objext=o
    # Xlf does not need/want an "_" on the names of fortran  
    # functions.  This var is used in fspec.pl.in in order not to
    # add the _ to the cname of the function.
    FORTRAN_UNDERSCORE=none

    # The IBM compilers don't use LDRUN_PREFIX.  By making it "-L"
    # the lib path will be repeated twice, but since the makefile
    # is set up assuming you need LDRUN_PREFIX, this is the easiest fix.
    LDRUN_PREFIX="-L"

    # xlc doesn't seem to like the <>...
    # KCC doesn't seem to like the <>...
    # g++ doesn't seem to like <> on the IBM either.
    TEMPLATE_BOX=

    case $CXX in
      *g++*)
         SOFLAGS="-shared ${SOFLAGS}"
         # bigtoc helps keep gcc from breaking...
         LDFLAGS="-Wl,-bbigtoc $LDFLAGS"
         ;;
      *)
#      for these compilers: *xlC*) or *KCC*)
         if test "$enable_64bit" = "yes"; then
            BIT64FLAG=-q64
         fi
         CFLAGS="-qrtti=all $BIT64FLAG $CFLAGS"

         # Allows for better optimization and removes INFORMATION message.
         # I think this is necessary for both KCC and xlC...
         if test "$optimize" = "yes"; then
            CFLAGS="-qMAXMEM=4096 $CFLAGS"
         fi
         ;;
    esac

    CFLAGS="$CFLAGS $extra_cflags"

    FFLAGS="$extra_cflags"
    CXXFLAGS="$CFLAGS"

    CPP="$CC -c"
    CXXCPP="$CXX -c"

    ;;
  *-darwin*)
    ac_objext=o
    LDRUN_PREFIX="-L"
    CFLAGS="$CFLAGS $extra_cflags -fno-common -dynamic -fcoalesce-templates -Wno-long-double "
    CXXFLAGS="$CXXFLAGS $CFLAGS"
    SCI_CHECK_VERSION($CC,--version,3.3,ver33=yes,ver33=no)
    if test $ver33 == no; then
	SOFLAGS="-dynamiclib -init __ZNSt8ios_base4InitC4Ev \${LIBDIR}/../osx/osx_init.o  ${SOFLAGS}"
	NEED_OSX_HACK="yes"
    else
	SOFLAGS="-dynamiclib"
	NEED_OSX_HACK="no"

    fi
    AC_SUBST(NEED_OSX_HACK)
    ;;

  *)
    # linux
    ac_objext=o
    BASE_LIB_DIR=lib
    CFLAGS=" -Wall"
    if test "$CC" != "pgcc"; then
      SCI_CHECK_VERSION($CC,--version,2.95.3,,
        echo configure: error: wrong $CC version.; exit)
      if test $sci_check_fortran = "yes"; then
        SCI_CHECK_VERSION($F77,--version,0.5.25,,
          echo configure: error: wrong $F77 version.; exit)
      fi
      SCI_CHECK_VERSION($CXX,--version,2.95.3,,
        echo configure: error: wrong $CXX version.; exit)
    fi
    AC_PROG_CPP
    AC_PROG_CXXCPP
    AC_CHECK_PROG(AS,as,as)
    AC_CHECK_PROG(LD,ld,ld)
    WOFF=" -w"
    CPPFLAGS="$CPPFLAGS $WOFF"
    CFLAGS="$CFLAGS $extra_cflags"

    case $F77 in
      *g77*)
         # extra_fflags has array checking if in debug mode.
         FFLAGS="$FFLAGS $extra_cflags $extra_fflags"

         # put '_' at end and two '_'s if there is a '_' in name. 
         # This flag is used by the genfspec.pl script in Uintah (generated
         # from fspec.pl.in).
	 FORTRAN_UNDERSCORE=linux
         ;;
      *)
         FFLAGS="$FFLAGS $extra_cflags"
         ;;
    esac

    CXXFLAGS="$CXXFLAGS $extra_cflags"
    SOFLAGS="-shared ${SOFLAGS}"
    NEED_SONAME=yes
    ;;
esac

AC_MSG_CHECKING($CC dependency regen switch)
rm -f configure-test.*
rm -rf configure-test-dir
mkdir configure-test-dir
echo "#include <stdio.h>" > configure-test-dir/configure-test.cc

# Try to automatically determine these three things
# One of twopass, move or normal
CC_DEPEND_MODE=unknown
# Compiler-specific flag
CC_DEPEND_REGEN=unknown
# Filename extension
CC_DEPEND_EXT=unknown

# Try -MDupdate
for flag in -MDupdate; do
  if test $CC_DEPEND_MODE = unknown; then
    rm -f configure-test.o configure-test-dir/configure-test.o
    ac_try="$CXX $CFLAGS -c $flag configure-test-dir/configure-test.d -o configure-test-dir/configure-test.o configure-test-dir/configure-test.cc >/dev/null 2>conftest.out"
    (eval $ac_try) 2>&5
    if test -f configure-test-dir/configure-test.d ; then
      CC_DEPEND_MODE=normal
      CC_DEPEND_REGEN='-MDupdate $(dir $@)depend.mk'
      CC_DEPEND_EXT=depend.mk
    fi
  fi
done

# Because xlC ignores anything after the -M (ie: -Mgarbage or -Mupdate
# is the same as -M) and creates a .u file anyway, to make the
# following test work, we need to get rid of the .u file and carry on:
rm -f configure-test-dir/configure-test.u

# Try -MD or -M
if test $CC_DEPEND_MODE = unknown; then
  for flag in -MD -M; do
    for ext in d u o.d; do
      rm -f configure-test.o configure-test-dir/configure-test.o
      ac_try="$CXX $CFLAGS -c $flag configure-test-dir/configure-test.cc -o configure-test-dir/configure-test.o >/dev/null 2>conftest.out"
      (eval $ac_try) 2>&5
      if test -f configure-test-dir/configure-test.$ext ; then
	CC_DEPEND_REGEN=$flag
	CC_DEPEND_EXT=$ext
        if grep "^configure-test-dir/configure-test.o:" configure-test-dir/configure-test.$ext > /dev/null; then
          CC_DEPEND_MODE=normal
          break 2
        elif grep "^configure-test.o:" configure-test-dir/configure-test.$ext > /dev/null; then
          CC_DEPEND_MODE=modify
          break 2
        fi
      fi
      if test -f configure-test.$ext; then
	CC_DEPEND_REGEN=$flag
	CC_DEPEND_EXT=$ext
        if grep "^configure-test-dir/configure-test.o:" configure-test.$ext > /dev/null; then
          CC_DEPEND_MODE=move
        elif grep "^configure-test.o:" configure-test-dir/configure-test.$ext > /dev/null; then
          CC_DEPEND_MODE=modify2
        fi
        break 2
      fi
    done
  done
fi

# Last resort, try one that spits out to stdout
if test $CC_DEPEND_MODE = unknown; then
  for flag in -M; do
    rm -f configure-test.o configure-test-dir/configure-test.o
    ac_try="$CXX $CFLAGS -c $flag configure-test-dir/configure-test.cc >configure-test-dir/configure-test.d 2>conftest.out"
    (eval $ac_try) 2>&5
    if test -s configure-test-dir/configure-test.d ; then
      CC_DEPEND_MODE=stdout
      CC_DEPEND_REGEN=$flag
      CC_DEPEND_EXT=d
      break
    fi
  done
fi

if test $CC_DEPEND_MODE = unknown; then
  AC_MSG_ERROR(Cannot determine dependency generation flag)
fi

AC_MSG_RESULT([$CC_DEPEND_REGEN (mode: $CC_DEPEND_MODE, ext: $CC_DEPEND_EXT)])
rm -f configure-test.*
rm -rf configure-test-dir

F77_DEPEND_REGEN=$CC_DEPEND_REGEN
F77_DEPEND_MODE=$CC_DEPEND_MODE
F77_DEPEND_EXT=$CC_DEPEND_EXT
AC_SUBST(CC_DEPEND_MODE)
AC_SUBST(CC_DEPEND_REGEN)
AC_SUBST(CC_DEPEND_EXT)
AC_SUBST(F77_DEPEND_MODE)
AC_SUBST(F77_DEPEND_REGEN)
AC_SUBST(F77_DEPEND_EXT)

# Other programs
AC_PATH_PROG(GMAKE,gmake,NOT_FOUND,,)
if test "$GMAKE" = "NOT_FOUND"; then
  AC_PATH_PROG(MAKE,make,NOT_FOUND,,)
else
  MAKE=$GMAKE
fi
if test "$MAKE" = "NOT_FOUND"; then
  SCI_MSG_ERROR(GNU make not found.)
fi

AC_SUBST(MAKE)

SCI_CHECK_VERSION($MAKE,--version,3.78.1,,
  echo configure: error: wrong GNU make version.; exit)

AC_PROG_AWK

## test for bison. yacc is not sufficient as it has no -o option.
AC_CHECK_PROG(YACC, bison, bison, nobison, $PATH /usr/bin)
if test "$YACC" = "nobison"; then
  echo "configure: error: bison is required, and was not found in your PATH."
  echo "                  Please add it to your PATH, or set the environment"
  echo "                  variable YACC to the appropriate bison executable,"
  echo "                  then run configure again."
  exit
fi

AC_PROG_LEX
LEX_LIB_FLAG="$LEXLIB"
AC_SUBST(LEX_LIB_FLAG)

if test "$LEX" != "flex" -a "$LEX" != "lex"; then
  echo "configure: error: flex or lex is required, and was not found in your PATH."
  echo "                  Please add it to your PATH, or set the environment"
  echo "                  variable LEX to the appropriate lex executable,"
  echo "                  then run configure again."
  exit
fi

AC_CHECK_PROG(ETAGS, etags, etags)

if test "$sci_check_perl" = "yes" ; then
  AC_CHECK_PROG(PERL, perl, perl, 
    echo configure: error: perl was not found in your PATH.;exit)
else
  AC_CHECK_PROG(PERL, perl, perl)
fi

if test "$PERL" = "perl"; then
  PERL=`type perl | sed 's%^.* %%'`
fi

##  Create the Packages/Uintah/tools dir for later use.
if test "$sci_check_tools" = "yes" ; then
  if test ! -d "Packages/Uintah/tools"; then
    AC_MSG_RESULT(Creating Packages/Uintah/tools directory)
    mkdir -p "Packages/Uintah/tools"
  fi
  if test "$ac_cv_f77_compiler_gnu" = "yes" ; then
     G77=1
  else
     G77=0
  fi
fi

AC_SUBST(FORTRAN_UNDERSCORE)
AC_SUBST(G77)
AC_SUBST(CC)
AC_SUBST(F77)
AC_SUBST(CXX)
AC_SUBST(AS)
AC_SUBST(LD)
AC_SUBST(CXXFLAGS)
AC_SUBST(CPPFLAGS)
AC_SUBST(FFLAGS)
AC_SUBST(LDFLAGS)
AC_SUBST(SOFLAGS)
AC_SUBST(ASFLAGS)
AC_SUBST(LDRUN_PREFIX)
AC_SUBST(extra_cflags)
AC_SUBST(MAKE_PARALLELISM)
AC_SUBST(NEED_SONAME)
AC_SUBST(PERL)
AC_SUBST(REPOSITORY_FLAGS)
AC_SUBST(TEMPLATE_TAG)
AC_SUBST(TEMPLATE_BOX)

##  set the default language after the build tools are found
AC_LANG([C])
##  ----------  See if this is a 64 bit machine --------------------------

MY_TEMP=$CPPFLAGS

AC_CHECK_SIZEOF(void*)

case $ac_cv_sizeof_voidp in
 8)
  NBITS=64
  ;;
 4)
  NBITS=32
  ;;
 *)
  SCI_MSG_ERROR([
  This machine has some strange sizeof(void*): $ac_cv_sizeof_void_p.
  Usually this means that your LD_LIBRARY_PATH is set incorrectly for the 
  compiler you are using.  To verify that this is the problem you will need 
  to copy the C++  program from the end of the config.log file and run the 
  compile command which is just before it in config.log.  Run the executable 
  and see what error message you get. You can also just fix your 
  LD_LIBRARY_PATH if you know what is wrong with it.
  Exiting due to size of void* problem! Read above warnings!])
  ;;
esac
AC_SUBST(NBITS)

AC_LANG([C++])
#########################################################################
## Make sure number of bit build for Thirdparty and SCIRun compile match
##

if test "$thirdparty" && test -e $thirdparty; then
  if test "$enable_64bit" = "yes"; then
    case "$thirdparty" in 
      *32bit*)
	AC_MSG_ERROR(You specified 64 bit compile but have 32 bit Thirdparty?)
      ;;
    esac
  else
    case "$thirdparty" in 
      *64bit*)
	AC_MSG_ERROR(You specified 32 bit compile but have 64 bit Thirdparty?)
      ;;
    esac
  fi

  sci_thirdparty_dir="$thirdparty"
else
  if test "$thirdparty"; then
    SCI_MSG_ERROR(Specified thirdparty directory: $thirdparty does not exist!)
  fi
fi

##  sci_thirdparty_dir is the path to the current thirdparty
##  if thirdparty not set use defaults

if test -z "$thirdparty"; then
  sci_tp_version=1.10
  sci_tp_arch=`uname -s`
  sci_compiler_vers=gcc-3.0.4
  ac_try="$CC -v > /dev/null 2> conftest.out"
  (eval $ac_try) 2>&5
  vers=`grep "gcc version" conftest.out | cut -d' ' -f 3`
  rm -f conftest.out
  if test -n "$vers"; then
    sci_compiler_vers=gcc-$vers
  else
    ac_try="$CC -version > /dev/null 2> conftest.out"
    (eval $ac_try) 2>&5
    vers=`grep MIPSpro conftest.out | cut -d' ' -f 4`
    rm -f conftest.out
    if test -n "$vers"; then
      sci_compiler_vers=MIPSpro-$vers
    else
      sci_compiler_vers=""
    fi
  fi

  # first if we are in the sci facility
  _sci_tp_base=/usr/sci/projects
  _sci_tp_subdir=$sci_tp_arch/$sci_compiler_vers-${NBITS}bit
  _sci_tp_path=$_sci_tp_base/SCIRun/Thirdparty/$sci_tp_version/$_sci_tp_subdir

  if test -e $_sci_tp_path; then
    sci_thirdparty_dir="$_sci_tp_path"
  else
    # now try the default install location for thirdparty
    _sci_tmp_tp_path=$_sci_tp_path
    _sci_tp_base=/usr/local
    _sci_tp_path=$_sci_tp_base/SCIRun/Thirdparty/$sci_tp_version/$_sci_tp_subdir
    if test -e $_sci_tp_path; then
      sci_thirdparty_dir="$_sci_tp_path"
    else
      AC_MSG_ERROR(Thidparty installation does not exist in $_sci_tmp_tp_path or in $_sci_tp_path. Please specify the installation like this --with-thirdparty=DIR)
    fi
  fi

  # extra thirdparty code for CCA
  cca_tp_version=1.0.0
  cca_tp_path=$_sci_tp_base/CCA/Thirdparty/$cca_tp_version/$_sci_tp_subdir

  # extra thirdparty code for Uintah
  uintah_tp_version=1.0.0
  uintah_tp_path=$_sci_tp_base/Uintah/Thirdparty/$uintah_tp_version/$_sci_tp_subdir

  if test "$pkg_uintah" = "yes"; then
    if test -e $cca_tp_path && test -e $uintah_tp_path; then
      sci_thirdparty_dir="$sci_thirdparty_dir $cca_tp_path $uintah_tp_path"
    else
       AC_MSG_WARN(Default Uintah thirdparty directory does not exists!)
    fi
  fi

  # extra thirdparty code for rtrt
  rtrt_tp_version=1.0.0
  rtrt_tp_path=$_sci_tp_base/rtrt/Thirdparty/$rtrt_tp_version/$_sci_tp_subdir

  if test "$pkg_rtrt" = "yes"; then
    if test -e $rtrt_tp_path; then
      sci_thirdparty_dir="$sci_thirdparty_dir $rtrt_tp_path"
    else
      AC_MSG_WARN(Default rtrt thirdparty directory $rtrt_tp_path does not exists!)
    fi
  fi

  AC_MSG_RESULT(Using default thirdparty directory $sci_thirdparty_dir)
fi

for i in $sci_thirdparty_dir; do
  if test ! -d "$i"; then
    continue
  fi

  #this is where the lib actually lives
  if test ! -d "$i/lib"; then
    continue
  fi
  # make sure there is no leading space in path
  if test -z "$sci_thirdparty_libs"; then
    sci_thirdparty_libs="$i/lib"
  else
    sci_thirdparty_libs="$sci_thirdparty_libs $i/lib"
  fi
  if test -z "$sci_thirdparty_includes"; then
    sci_thirdparty_includes="$i/include"
  else
    sci_thirdparty_includes="$sci_thirdparty_includes $i/include"
  fi
done

#########################################################################
##  Set the no_optimize_flag 
##  

case $host in
  *-irix*) 
    no_optimize_flag=-O0
    TIME_IMPL=Time_irix.cc
    ;;
  *linux*)
    no_optimize_flag=-O0
    ;;
  *-aix*)

    AC_MSG_WARN(AIX is not completely supported!)

    # On AIX, don't automatically generate .so's.  Make .a's and then
    # later we will make .so's from them.
    SO_OR_ARCHIVE_EXTENSION=a
    no_optimize_flag=-O0
    case $CXX in
      *xlC*)
         # xlC does not know the -O0 flag.
         no_optimize_flag=""
	 ;;
      *KCC*)
         ;;
      *g++*)
	 ;;
      *)
         AC_MSG_WARN(Do not know this compiler.)
         AC_MSG_WARN(Defaulting to -O0 for the no optimize flag.)
         ;;
    esac
    ;;
  *)
    ;;
esac

##  ----------------------------------------------------------------------
##  ----------  required checks for required libraries and headers  ------
##  ----------------------------------------------------------------------

echo 
AS_MESSAGE(checking for required standard components......)
echo

##  --  search for the math library  -------------------------------------
AC_LANG(C)

SCI_TRY_LINK(M,math libraries and headers,[
#include <math.h>
],,m,,,[
  double d = fabs(69.05);
],not-optional)

AC_SUBST(M_LIB_FLAG)
AC_SUBST(M_LIB_DIR_FLAG)

##  --  search for X  ----------------------------------------------------
if test -z "$with_X11_lib"; then
  _x_lib_dir="/usr/X11R6/lib"
  _x_inc_dir="/usr/X11R6/include"
else 
  _x_lib_dir="$with_X11_lib"
  _x_inc_dir="$with_X11_lib/../include"
fi

SCI_TRY_LINK(X,X libraries and headers, [
#include <X11/Xlib.h>
],$_x_inc_dir,X11 Xext Xaw Xi Xt Xmu,$_x_lib_dir,,,not-optional)


AC_SUBST(X_LIB_FLAG)
AC_SUBST(X_LIB_DIR_FLAG)

##  --  check for the required thread library  ---------------------------

case $host in
  *-irix*)
    if test "$threads" = "unknown"; then
      threads=irix
    elif test "$threads" != "pthreads" -a "$threads" != "irix"; then
      SCI_MSG_ERROR(unknown thread type for host $host: $threads. exiting.)
    fi
    ;;
  *)
    if test "$threads" = "unknown"; then
      threads=pthreads
    elif test "$threads" != "pthreads"; then
      SCI_MSG_ERROR(unknown thread type for host $host: $threads. exiting.)
    fi
    ;;
esac

case $threads in
   irix)
      SCI_TRY_LINK(FETCHOP,fetchop library,,,fetchop,,,,not-optional)
      if test "$HAVE_FETCHOP" = "yes"; then
         CFLAGS="$CFLAGS -DSTL_SGI_THREADS"
         THREAD_LIB_FLAG="$FETCHOP_LIB_FLAG"
         DEF_THREAD=
      fi
      ;;
   pthreads)
      SCI_TRY_LINK(PTHREAD,pthread library and headers,[
        #include <pthread.h>
        #include <semaphore.h>
        ],,pthread,,,,not-optional)

      if test "$HAVE_PTHREAD" = "yes"; then
        THREAD_LIB_FLAG="$PTHREAD_LIB_FLAG"
        DEF_THREAD="#define SCI_PTHREAD 1"
      fi

      AC_MSG_CHECKING(for semaphore library)
      old_LIBS="$LIBS"
      SEMAPHORE_LIB_FLAG=unknown
      for i in "" "$THREAD_LIB_FLAG" -lrt; do
        LIBS="$old_LIBS $i"
        AC_TRY_LINK([
#include <semaphore.h>
      ], [
    sem_t sem;
    sem_init(&sem, 0, 0);
    sem_post(&sem);
      ], [
         SEMAPHORE_LIB_FLAG=$i
         break
         ])
      done
      if test "$SEMPAPHORE_LIB_FLAG" = unknown; then
        AC_MSG_RESULT(not found)
        AC_MSG_ERROR([semaphore library(sem_post, sem_wait, etc) needed for pthreads])
      else
        if test -z "$SEMAPHORE_LIB_FLAG"; then
          AC_MSG_RESULT(none required)
        elif test "$SEMAPHORE_LIB_FLAG" = "$THREAD_LIB_FLAG"; then
          AC_MSG_RESULT(already in thread library: $THREAD_LIB_FLAG)
          SEMAPHORE_LIB_FLAG=""
        else
          AC_MSG_RESULT($SEMAPHORE_LIB_FLAG)
        fi
      fi
      LIBS="$old_LIBS"
      ;;
   *)
      SCI_MSG_ERROR(unknown thread type for host $host: $threads. exiting.)
      ;;
esac      

THREAD_IMPL="Thread_$threads.cc"
AC_SUBST(THREAD_IMPL)

AC_SUBST(THREAD_LIB_FLAG)
AC_SUBST(DEF_THREAD)

AC_SUBST(SEMAPHORE_LIB_FLAG)

AC_SUBST(CFLAGS)

##  --  search for OpenGL  -----------------------------------------------

if test "$with_opengl"; then
  _sci_lib="$with_opengl/lib"
  _sci_inc="$with_opengl/include"
else
  _sci_lib="/usr/X11R6/lib"
  _sci_inc="/usr/X11R6/include"
fi

case $host in
  *linux*)
  GLU_OTHERS="pthread"
  ;;
  *)
  GLU_OTHERS=""
  ;;
esac

SCI_TRY_LINK(GL,OpenGL libraries and headers,[
#include<GL/gl.h>

#ifndef GLAPIENTRY
  #define GLAPIENTRY
#endif

#ifndef GLAPI
  #define GLAPI
#endif

#include<GL/glu.h>
#include<GL/glx.h>

],$_sci_inc,GLU GL $GLU_OTHERS,$_sci_lib,,,not-optional)

AC_SUBST(GL_LIB_FLAG)
AC_SUBST(GL_LIB_DIR_FLAG)
AC_SUBST(INC_GL_H)


##  --  search for dlopen  -----------------------------------------------
SCI_TRY_LINK(DL,dlopen in libc,[
#include <dlfcn.h>
],,c,,,[
  dlopen("bogus", 0);
],optional)

if test "$HAVE_DL" = "yes"; then
   HAVE_LIB_DL="yes"
else
  case $host in
  *-darwin*)
    SCI_TRY_LINK(DL,dlopen in libdl,[
    #include <dlfcn.h>
    ],$thirdparty/include,dl,$thirdparty/lib,,[
      dlopen("bogus", 0);
    ],not-optional)
    ;;
  *)
    SCI_TRY_LINK(DL,dlopen in libdl,[
    #include <dlfcn.h>
    ],,dl,,,[
      dlopen("bogus", 0);
    ],not-optional)
    ;;
  esac
fi

AC_SUBST(DL_LIB_FLAG)
AC_SUBST(DL_LIB_DIR_FLAG)

##  --  search for required headers  --------------------------------------

##  --  required standard headers
SCI_TRY_LINK(STD_HDRS,required standard headers,[
#include <stdio.h> 
#include <stdlib.h> 
#include <stddef.h> 
#include <stdarg.h> 
#include <string.h> 
#include <limits.h>
#include <errno.h> 
#include <fcntl.h> 
#include <float.h> 
#include <memory.h> 
#include <unistd.h> 
#include <netdb.h>
#include <ctype.h> 
#include <signal.h> 
#include <locale.h> 
#include <dirent.h> 
#include <assert.h> 
#include <sys/select.h>
#include <sys/types.h> 
#include <sys/time.h> 
#include <sys/stat.h> 
#include <sys/socket.h> 
#include <sys/timeb.h>
#include <sys/mman.h> 
#include <sys/param.h> 
#include <sys/ioctl.h> 
#include <sys/resource.h> 
#include <sys/wait.h>
#include <sys/utsname.h> 
#include <rpc/types.h> 
#include <netinet/in.h>
],,,,,,not-optional)

# don't test for malloc.h or values.h on darwin
case $host in
  *-darwin*)
    ;;
  *-irix*)
    # test for required sqi headers as well
    SCI_TRY_LINK(STD_HDRS,required standard headers,[
      #include <ieeefp.h> 
      #include <ulocks.h>
      #include <values.h>
      #include <malloc.h>
      ],,,,,,not-optional)
    ;;
  *)
    SCI_TRY_LINK(STD_HDRS,required standard headers,[
      #include <values.h>
      #include <malloc.h>
      ],,,,,,not-optional)
    ;;
esac


##  --  required STL headers 
##  --      (only look for STL after going into cplusplus mode)
##
AC_LANG(C++)

SCI_TRY_LINK(STL_HEADERS,standard template library headers,[
#include <algorithm>
#include <vector>
#include <map> 
#include <iostream>
#include <string>
],,,,,[
  std::vector<int> vec;
  std::cout << "blah" << std::endl;
],not-optional)

AC_MSG_CHECKING(for location of hash_map)
DEF_HASHMAP=None
AC_TRY_COMPILE([#include <hash_map>
                std::hash_map<int, int> xx;], ,
		[DEF_HASHMAP=HAVE_STD_HASHMAP])
if test "$DEF_HASHMAP" = "None"; then
  AC_TRY_COMPILE([#include <ext/hash_map>
                 std::hash_map<int, int> xx;], ,
		 [DEF_HASHMAP=HAVE_EXT_HASHMAP])
fi
if test "$DEF_HASHMAP" = "None"; then
  AC_TRY_COMPILE([#include <ext/hash_map>
                 __gnu_cxx::hash_map<int, int> xx;], ,
                 [DEF_HASHMAP=HAVE_GNU_HASHMAP])
fi
if test "$DEF_HASHMAP" = "None"; then
  DEF_HASHMAP=
  AC_MSG_RESULT(not found)
  AC_MSG_WARN([Cannot find hash_map, will use map instead])
else
  AC_MSG_RESULT(found: $DEF_HASHMAP)
  DEF_HASHMAP="#define $DEF_HASHMAP"
fi
AC_SUBST(DEF_HASHMAP)
AC_CHECK_HEADERS(ext/algorithm)

##  --  search for tcl stuff  ---------------------------------------------
echo 
AC_CHECKING(for required Tcl components...)
echo

if test -z "$with_tcl"; then
  _sci_inc="$sci_thirdparty_includes"
  _sci_lib="$sci_thirdparty_libs"
else
  _sci_inc="$with_tcl/include"
  _sci_lib="$with_tcl/lib"
fi

if test -f "$_sci_lib/tclConfig.sh"; then 
  . $_sci_lib/tclConfig.sh; 
  _sci_tcl_inc="$_sci_inc $TCL_SRC_DIR/generic"
fi

SCI_TRY_LINK(TCL,required tcl library and headers,[
#include <tcl.h> 
#include <tclPort.h>
#include <tclMath.h>
#include <tclInt.h>
],$_sci_tcl_inc,tcl,$_sci_lib,,,not-optional)

AC_SUBST(LIB_DIR_TCL)
AC_SUBST(TCL_LIB_FLAG)
AC_SUBST(TCL_LIB_DIR_FLAG)
AC_SUBST(INC_TCL_H)

# -- tk --
if test -f "$_sci_lib/tkConfig.sh"; then 
  . $_sci_lib/tkConfig.sh; 
  _sci_tk_inc="$_sci_inc $TK_SRC_DIR/generic"
fi

#
# Note: TK_DEFS (used below to help resolve tk issues) comes from tkConfig.sh
#
SCI_TRY_LINK(TK,required tk library and headers,[
#include <tk.h> 
#include <tk3d.h>
#include <tkPort.h>
],$_x_inc_dir $_sci_tk_inc,tk,$_sci_lib,$TK_DEFS $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,not-optional)

AC_SUBST(TK_LIB_FLAG)
AC_SUBST(TK_LIB_DIR_FLAG)
AC_SUBST(INC_TK_H)

# -- itcl --
SCI_TRY_LINK(ITCL,required itcl library and headers,[
#include <itcl.h> 
],$_sci_inc,itcl,$_sci_lib,$TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,not-optional)

AC_SUBST(LIB_DIR_ITCL)
AC_SUBST(ITCL_LIB_FLAG)
AC_SUBST(ITCL_LIB_DIR_FLAG)
AC_SUBST(INC_ITCL_H)

# -- itk --
SCI_TRY_LINK(ITK,required itk library and headers,[
#include <itk.h> 
],$_x_inc_dir $_sci_inc,itk,$_sci_lib,$ITCL_LIB_DIR_FLAG $ITCL_LIB_FLAG $TK_LIB_DIR_FLAG $TK_LIB_FLAG $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,not-optional)

AC_SUBST(ITK_LIB_FLAG)
AC_SUBST(ITK_LIB_DIR_FLAG)
AC_SUBST(INC_ITK_H)

# -- blt --
SCI_TRY_LINK(BLT,required BLT library and headers,[
#include <blt.h> 
],$_sci_inc,BLT,$_sci_lib,$TK_LIB_DIR_FLAG $TK_LIB_FLAG $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,not-optional)

AC_SUBST(BLT_LIB_FLAG)
AC_SUBST(BLT_LIB_DIR_FLAG)
AC_SUBST(INC_BLT_H)

if test "$pkg_CollabVis" = "yes"; then
  DEF_COLLAB_VIS="#define HAVE_COLLAB_VIS 1"
  HAVE_COLLAB_VIS="yes"
fi

AC_SUBST(DEF_COLLAB_VIS)
AC_SUBST(HAVE_COLLAB_VIS)


if test "$sci_check_plplot" = "yes"; then
  if test "$with_plplot"; then
    _sci_lib="$with_plplot/lib"
    _sci_inc="$with_plplot/include"
  else
    _sci_lib="/usr/local/plplot/lib $sci_thirdparty_libs"
    _sci_inc="/usr/local/plplot/include $sci_thirdparty_includes"
  fi


#  SCI_TRY_LINK(PLPLOTTCLTK,optional optimized plplottcltk library and headers,[
#
#  ],$_sci_inc,plplottcltk, $_sci_lib,$TK_LIB_DIR_FLAG $TK_LIB_FLAG $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,optional)


#  if test "$HAVE_PLPLOTTCLTK" = "no"; then
#    SCI_TRY_LINK(PLPLOTTCLTK,optional debug plplottcltk library and headers,[
#
#    ],$_sci_inc,plplottcltkd, $_sci_lib,$TK_LIB_DIR_FLAG $TK_LIB_FLAG $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,optional)
#  fi

  SCI_TRY_LINK(PLPLOT,optional optimized plplot library and headers,[
  #include <plplot/plplot.h> 
  ],$_sci_inc,plplot, $_sci_lib,$TK_LIB_DIR_FLAG $TK_LIB_FLAG $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,optional)

  if test "$HAVE_PLPLOT" = "no"; then
    SCI_TRY_LINK(PLPLOT,optional debug plplot library and headers,[
    #include <plplot/plplot.h> 
    ],$_sci_inc,plplotd, $_sci_lib,$ITK_LIB_DIR_FLAG $ITK_LIB_FLAG $ITCL_LIB_DIR_FLAG $ITCL_LIB_FLAG,,optional)
  fi

  if test "$HAVE_PLPLOT" = "yes"; then
    DEF_PLPLOT="#define HAVE_PLPLOT 1"
  fi

  SCI_TRY_LINK(TCLMATRIX,optional optimized tclMatrix library and headers,[
  #include <plplot/tclMatrix.h> 
  ],$_sci_inc,tclmatrix, $_sci_lib,$TK_LIB_DIR_FLAG $TK_LIB_FLAG $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,optional)

  if test "$HAVE_TCLMATRIX" = "no"; then
    SCI_TRY_LINK(TCLMATRIX,optional debug tclMatrix library and headers,[
    #include <plplot/tclMatrix.h> 
    ],$_sci_inc,tclmatrixd, $_sci_lib,$TK_LIB_DIR_FLAG $TK_LIB_FLAG $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,optional)
  fi

  if test "$HAVE_TCLMATRIX" = "yes"; then
    DEF_TCLMATRIX="#define HAVE_TCLMATRIX 1"
  fi
fi #end sci_check_plplot

#AC_SUBST(PLPLOTTCLTK_LIB_FLAG)
#AC_SUBST(PLPLOTTCLTK_LIB_DIR_FLAG)
AC_SUBST(PLPLOT_LIB_FLAG)
AC_SUBST(PLPLOT_LIB_DIR_FLAG)
AC_SUBST(INC_PLPLOT_H)
AC_SUBST(DEF_PLPLOT)

AC_SUBST(TCLMATRIX_LIB_FLAG)
AC_SUBST(TCLMATRIX_LIB_DIR_FLAG)
AC_SUBST(INC_TCLMATRIX_H)
AC_SUBST(DEF_TCLMATRIX)

CORRECT_TCL_VERSION=no
CORRECT_TK_VERSION=no
CORRECT_ITCL_VERSION=no

SCI_CHECK_VAR_VERSION(Tcl,$TCL_VERSION$TCL_PATCH_LEVEL,8.3.2,,continue)
SCI_CHECK_VAR_VERSION(Tk,$TK_VERSION$TK_PATCH_LEVEL,8.3.2,,continue)
SCI_CHECK_VAR_VERSION(equal Tcl and Tk, $TCL_VERSION$TCL_PATCH_LEVEL,
                      $TK_VERSION$TK_PATCH_LEVEL,,continue,-ne)

AC_SUBST(TCL_PREFIX)
AC_SUBST(TCL_VERSION)
AC_SUBST(TK_VERSION)

AC_SUBST(ITCL_WIDGETS)

##  --  search for xerces-c++ --
echo
AS_MESSAGE(checking for required Xerces-C++ components......)
echo

if test "$with_xercesc"; then
  _xer_lib_path="$with_xercesc/lib"
  _xer_inc="$with_xercesc/include"
else
  _xer_lib_path="$sci_thirdparty_libs"
  _xer_inc="$sci_thirdparty_includes"
fi

_xer_lib="xerces-c"
case $host in
  *aix*)
      # Needed on blue pacific for stricmp
      _xer_lib="xerces-c cfgdce"
      ;;
esac

SCI_TRY_LINK(XERCES_C,xerces library and headers,[
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp> 
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
],$_xer_inc,$_xer_lib,$_xer_lib_path,,[
  DOMNode *node = 0;
  node->getNodeName();
],not-optional)

AC_SUBST(XERCES_C_LIB_FLAG)
AC_SUBST(XERCES_C_LIB_DIR_FLAG)
AC_SUBST(INC_XERCESC_H)

##  ----------------------------------------------------------------------
##  ----------  required checks for optional libraries and headers  ------
##  ----------------------------------------------------------------------

##  --  search for optional headers  ------------------------------------

SCI_TRY_LINK(SSTREAM,sstream header,[
#include <sstream>
],,,,,,optional)

if test "$HAVE_SSTREAM" = "yes"; then
   SSTREAM_COMPAT="no"
else
   # Use SCIRun sstream compatibility files...
   SSTREAM_COMPAT="yes"
fi

AC_SUBST(SSTREAM_COMPAT)

##  --  search for SGI'S traceback library  ------------------------------ 

if test -z "$with_exc"; then
  _sci_lib=
  _sci_inc=
else
  _sci_lib="$with_exc"
  _sci_inc="$with_exc/../include"
fi

SCI_TRY_LINK(EXC,traceback library and headers,[
#include <libexc.h>
],$_sci_inc,exc,$_sci_lib,,,optional)

if test "$HAVE_EXC" = "yes"; then
  DEF_EXC="#define HAVE_EXC 1"
fi

if test "$with_exc" && test ! "$DEF_EXC"; then
  echo
  AC_MSG_ERROR(One or more of the EXC components is missing.)
  echo
fi

AC_SUBST(EXC_LIB_FLAG)
AC_SUBST(EXC_LIB_DIR_FLAG)
AC_SUBST(DEF_EXC)

##  --  search for mpeg  -------------------------------------------------

if test "$with_mpeg"; then
  _sci_lib="$with_mpeg/lib"
else
  _sci_lib="$sci_thirdparty_libs"
fi

SCI_TRY_LINK(MPEGE,MPEG library,,,mpege,$_sci_lib,,,optional)

if test "$HAVE_MPEGE"; then
  DEF_MPEG="#define HAVE_MPEG 1"
fi

if test "$with_mpeg" && test ! "$DEF_MPEG"; then
  echo
  AC_MSG_ERROR(One or more of the MPEG components is missing.)
  echo
fi

AC_SUBST(MPEGE_LIB_FLAG)
AC_SUBST(MPEGE_LIB_DIR_FLAG)
AC_SUBST(DEF_MPEG)

##  --  search for ImageMagick --------------------------------------

if test "$with_imagemagick"; then
  _sci_lib="$with_imagemagick"
else
  _sci_lib="$sci_thirdparty_dir"
fi

for i in $_sci_lib; do
  if test ! -d "$i"; then
    continue
  fi

  #this is where the lib actually lives
  if test ! -d "$i/lib"; then
    continue
  fi

  #this is where the scripts for configuration live
  if test ! -d "$i/bin"; then
    continue
  fi

  # capture in the libs to link
  if test -e $i/bin/Magick-config; then
    IM_LDFLAGS=`$i/bin/Magick-config --ldflags`
    IM_LIBS=`$i/bin/Magick-config --libs`
  else
    continue
  fi

  SCI_TRY_LINK(MAGICK,optional ImageMagick library and headers,[
  #include <magick/magick.h>
  ],$i/include,Magick,$i/lib,$IM_LDFLAGS $IM_LIBS,[
  WriteImage
  ],optional)

  if test "$HAVE_MAGICK"; then
    DEF_MAGICK="#define HAVE_MAGICK 1"
    MAGICK_EXTRA_LIB_FLAG="$IM_LDFLAGS"
    MAGICK_EXTRA_LIB_DIR_FLAG="$IM_LIBS"
    break
  fi
done
 
if test "$with_imagemagick" && test ! "$DEF_MAGICK"; then
  echo
  AC_MSG_ERROR(One or more of the ImageMagick components is missing.)
  echo
fi

AC_SUBST(MAGICK_LIB_FLAG)
AC_SUBST(MAGICK_LIB_DIR_FLAG)
AC_SUBST(MAGICK_EXTRA_LIB_FLAG)
AC_SUBST(MAGICK_EXTRA_LIB_DIR_FLAG)
AC_SUBST(INC_MAGICK_H)
AC_SUBST(DEF_MAGICK)

##  --  search for gzopen (and related)  ----------------------------------

if test "$with_zlib"; then
  _sci_lib="$with_zlib/lib"
  _sci_inc="$with_zlib/include"
else
  _sci_lib="$sci_thirdparty_libs"
  _sci_inc="$sci_thirdparty_includes"
fi

SCI_TRY_LINK(Z,required zlib library and header,[
#include <zlib.h>
],$_sci_inc,z,$_sci_lib,,,not-optional)

if test "$HAVE_Z"; then
   DEF_Z="#define HAVE_ZLIB 1"
fi

if test "$with_zlib" && test ! "$DEF_Z"; then
  echo
  AC_MSG_ERROR(One or more of the GZOPEN components is missing.)
  echo
fi

AC_SUBST(Z_LIB_FLAG)
AC_SUBST(Z_LIB_DIR_FLAG)
AC_SUBST(INC_Z_H)
AC_SUBST(DEF_Z)

##  --  search for the fortran library  ----------------------------------
if test $sci_check_fortran = "yes"; then
  F_LIB_FLAG=""
  case $host in 
    *-irix*)
      SCI_TRY_LINK(F,required fortran libraries,,,ftn fortran,,,,not-optional)
      ;;
    *aix*)
      # AIX Fortran Library Check
      SCI_TRY_LINK(F,required fortran libraries,,,xlf90,,,,not-optional)
      ;;
    *linux*)
      # Linux Library Check
      if test "$F77" = "pgf77"; then
        SCI_TRY_LINK(F,required fortran libraries,,,pgftnrtl,,,,not-optional)
      else
        SCI_TRY_LINK(F,required fortran libraries,,,g2c,,,,not-optional)
      fi
      ;;
    *dec*)
      SCI_TRY_LINK(F,required fortran libraries,,,for Ufor,,,,not-optional)
      ;;
    *-darwin*)
      # OSX Library Check
      #SCI_TRY_LINK(F,required fortran libraries,,,g2c,/sw/lib,,,not-optional)
      ;;
    *)
      SCI_TRY_LINK(F,required fortran libraries,,,F77,,,,not-optional)
      ;;
  esac
fi  
AC_SUBST(F_LIB_FLAG)

##  --  search for blas stuff  -------------------------------------------

if test "$sci_check_blas" = "yes"; then
  if test "$with_blas"; then
    _sci_lib="$with_blas"
  else
    _sci_lib="$sci_thirdparty_libs"
  fi
  
  case $host in
    *aix*)
       SCI_TRY_LINK(BLAS,optional essl library,,,
                      essl,$_sci_lib,$F_LIB_FLAG,,not-optional)
       ;;
    *dec*)
       SCI_TRY_LINK(BLAS,optional cxml library,,,
                      cxml,$_sci_lib,$F_LIB_FLAG,,not-optional)
       ;;
    *)
       SCI_TRY_LINK(BLAS,optional blas library,,,
                      blas,$_sci_lib,$F_LIB_FLAG,,optional)
       if test "$HAVE_BLAS" = "no"; then
         SCI_TRY_LINK(BLAS,optional fblas library,,,
                      fblas,$_sci_lib,$F_LIB_FLAG,,not-optional)
       fi
       ;;
  esac

  if test "$HAVE_BLAS" = "yes"; then
    DEF_BLAS="#define HAVE_BLAS 1"
  fi
fi # end sci_check_blas

AC_SUBST(BLAS_LIB_FLAG)
AC_SUBST(BLAS_LIB_DIR_FLAG)
AC_SUBST(DEF_BLAS)


##  --  search for lapack stuff  -----------------------------------------

if test "$sci_check_lapack" = "yes" &&
   test "$HAVE_F" && test "$HAVE_BLAS"; then
  if test "$with_lapack"; then
    _sci_lib="$with_lapack"
  else
    _sci_lib="$LIB_DIR_BLAS $sci_thirdparty_libs"
  fi


  SCI_TRY_LINK(LAPACK,optional lapack library,,,lapack,$_sci_lib,$BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG,,optional)

  if test "$HAVE_LAPACK" = "no"; then
    SCI_TRY_LINK(LAPACK,optional flapack library,,,flapack,$_sci_lib,$BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG,,optional)
  fi

  if test "$HAVE_LAPACK" = "no"; then
    SCI_TRY_LINK(LAPACK,optional complib.sgimath library,,,complib.sgimath,$_sci_lib,,,optional)
    if test "$HAVE_LAPACK" = "yes"; then
       AC_MSG_WARN(Using complib for lapack)
    fi
  fi

  if test "$HAVE_LAPACK" = "yes"; then
    DEF_LAPACK="#define HAVE_LAPACK 1"
  fi
fi

if test "$with_lapack" && test ! "$DEF_LAPACK"; then
  echo
  AC_MSG_ERROR(One or more of the LAPACK components is missing.)
  echo
fi

AC_SUBST(LAPACK_LIB_FLAG)
AC_SUBST(LAPACK_LIB_DIR_FLAG)

##  --  search for mpi  --------------------------------------------------

if test "$sci_check_mpi" = "yes"; then
  echo
  AS_MESSAGE(checking for required MPI components......)
  echo

  case $host in
  #  *aix*)
        # On LLNL blue pacific, the "mp" compilers (scripts) automagically
        # include the paths to mpi stuff...  However, if you don't want 
        # them to automatically force you to run in poe (batch) mode, you
        # can comment this out and link in MPI directly yourself.
        # The resulting executable works in non-mpi mode which can be easier
        # to debug (and to get to run as you don't go through the poe system.)
        # It is possible that the executable will not work in MPI mode.
        #
        # AC_MSG_WARN(On blue pacific the mp compilers automatically specify mpi.  No check made.)
        #
        # The reason you have to specify xlC/KCC above, and change it to
        # mpxlC/mpKCC here is that configure uses the compiler to create test
        # programs to figure things out about the system.  If you use
        # the mpCC compiler, then the test programs will not run (as they
        # try to run in POE mode.)
        #
        # CC="mp$CC"
        # CXX="mp$CXX"
        # ;;
     *)
        if test "$with_mpi"; then
           _sci_lib="$with_mpi/lib"
           _sci_inc="$with_mpi/include"
        else
           _sci_lib="$sci_thirdparty_libs"
           _sci_inc="$sci_thirdparty_includes"
        fi
        # /usr/include/mpi is the standard location of mpi on some linux dists.
        SCI_TRY_LINK(MPI,mpi library and header,[
        #include <mpi.h>
        ],$_sci_inc /usr/include/mpi,mpi lam util pmpi,$_sci_lib,,,optional)

        if test "$HAVE_MPI" = "no"; then
           SCI_TRY_LINK(MPI,mpi library and header,[
           #include <mpi.h>
           ],$_sci_inc /usr/include/mpi,mpi,$_sci_lib,,,optional)

           if test "$HAVE_MPI" = "no"; then
              SCI_TRY_LINK(MPI,mpich library and header,[
              #include <mpi.h>
              ],$_sci_inc /usr/include/mpi,mpich pmpich,$_sci_lib,,,optional)

              if test "$HAVE_MPI" = "yes"; then
  	        HAVE_MPICH="yes"
#  	        LIB_DIR_MPI="$LIB_DIR_MPICH"
#                MPI_LIB_FLAG="$MPICH_LIB_FLAG"
#                MPI_LIB_DIR_FLAG="$MPICH_LIB_DIR_FLAG"
              fi
           fi
        fi
        
        if test "$HAVE_MPI" = "yes"; then
           if test "$HAVE_MPICH" = "yes"; then
             DEF_MPI="#define HAVE_MPICH 1"
           else
             DEF_MPI="#define HAVE_MPI 1"
           fi
        fi
  esac
  
  if test ! "$DEF_MPI"; then
     echo
     SCI_MSG_ERROR(one or more of the MPI components is missing.)
  fi
fi #end sci_check_mpi

AC_SUBST(MPI_LIB_FLAG)
AC_SUBST(MPI_LIB_DIR_FLAG)
AC_SUBST(INC_MPI_H)
AC_SUBST(DEF_MPI)
AC_SUBST(HAVE_MPI)

##  --  search for petsc stuff  ------------------------------------------
# a required check for an optional library for base scirun, due to that 
# fact that if MatrixSolver will utilize uni-petsc if it exists, and uni-petsc
# depends upon petsc - also check for mpi-uni as it most often exists in the 
# same directory

# We only use the optimized petsc lib even if we are building a debug SCIRun
#if test "$debug" = "yes"; then
#  PETSC_SUB=libg
#else
  PETSC_SUB=libO
#fi
PETSC_VERSION_DIR=petsc-2.1.1

case $host in
  *-irix*)
    if test "$enable_64bit" = "yes"; then
      PETSC_ARCH=IRIX64
    else
      PETSC_ARCH=IRIX32
    fi
    ;;
  *linux*)
    PETSC_ARCH=linux
    ;;
  *-dec*)
    if test "$petsc"; then
      SCI_MSG_ERROR(PETSC ARCH not set for Tru64.  Need to fix this.)
    else
      PETSC_ARCH=dec
    fi
    ;;
  *-aix*)
    PETSC_ARCH=rs6000_sp
    ;;
  *-darwin*)
    PETSC_ARCH=ppc
    ;;
  *)
    PETSC_ARCH=""
    ;;
esac

# check for mpi-uni if either --with-unipetsc or --with-mpiuni,
#   check here to look in petsc's subdirs.
##  --  search for mpi-uni  --------------------------------------------
# Again required for the following uni-petsc checks, that attempt to support
# the MatrixSolver module.

if test "$with_mpiuni"; then
  _sci_lib="$with_mpiuni/lib"
  _sci_inc="$with_mpiuni/include" 
elif test "$with_unipetsc"; then
  _sci_lib="$with_unipetsc/lib/$PETSC_SUB/$PETSC_ARCH"
  _sci_inc="$with_unipetsc/src/sys/src/mpiuni $with_unipetsc/bmake/$PETSC_ARCH" 
else
  _sci_lib="$sci_thirdparty_libs"
  _sci_inc="$sci_thirdparty_includes"
fi

SCI_TRY_LINK(MPIUNI,optional mpi-uni libraries and headers,[
#include <mpi.h>
],$_sci_inc,mpiuni,$_sci_lib,,,optional)

if test "$HAVE_MPIUNI" = "no"; then
   SCI_TRY_LINK(MPIUNICH,optional mpiunich and pmpiunich libraries,,
                  $_sci_inc,mpiunich pmpiunich,$_sci_lib,,,optional)

   if test "$HAVE_MPIUNICH" = "yes"; then
     LIB_DIR_MPIUNI="$LIB_DIR_MPIUNICH"
     MPIUNI_LIB_FLAG="$MPIUNICH_LIB_FLAG"
     MPIUNI_LIB_DIR_FLAG="$MPIUNICH_LIB_DIR_FLAG"
   fi
fi

if test "$HAVE_MPIUNI" = "yes"; then
   if test "$HAVE_MPIUNICH" = "yes"; then
     DEF_MPIUNI="#define HAVE_MPIUNICH 1"
   else
     DEF_MPIUNI="#define HAVE_MPIUNI 1"
   fi
fi

if test "$with_mpiuni" && test ! "$DEF_MPIUNI"; then
  echo
  AC_MSG_ERROR(One or more of the MPIUNI components is missing.)
  echo
fi


AC_SUBST(MPIUNI_LIB_FLAG)
AC_SUBST(MPIUNI_LIB_DIR_FLAG)
AC_SUBST(INC_MPIUNI_H)
AC_SUBST(DEF_MPIUNI)


if test -z "$PETSC_ARCH"; then
   AC_MSG_ERROR(PETSC_ARCH environment var is not set.  Please set.)
fi


# Make sure we have either mpi or mpi-uni, blas, and lapack
# It seems that under AIX and OSF, the libs that cover blas also cover lapack
if (test "$HAVE_MPI" = "yes" || test "$HAVE_MPIUNI" = "yes") && 
    test "$HAVE_BLAS" = "yes" && 
   (test "$PETSC_ARCH" = "rs6000_sp" || test "$PETSC_ARCH" = "dec" ||
    test "$HAVE_LAPACK" = "yes"); then

  _sci_lib=""
  _sci_inc=""

  if test "$with_petsc"; then
    _sci_lib="$with_petsc/lib/$PETSC_SUB/$PETSC_ARCH"
    _sci_inc="$with_petsc/bmake/$PETSC_ARCH $with_petsc/include"
  else
    # Modify _sci_dirs to have correct path information:
    for i in $sci_thirdparty_dir; do
      if test ! -d "$i"; then
        continue
      fi
  
      #this is where the lib actually lives
      if test ! -d "$i/$PETSC_VERSION_DIR/lib/$PETSC_SUB/$PETSC_ARCH"; then
        continue
      fi
  
      _sci_lib="$_sci_lib $i/$PETSC_VERSION_DIR/lib/$PETSC_SUB/$PETSC_ARCH"
      _sci_inc="$_sci_inc $i/$PETSC_VERSION_DIR/include $i/$PETSC_VERSION_DIR/bmake/$PETSC_ARCH"
    done  
  fi

  SCI_TRY_LINK(PETSC,optional petsc libraries and headers,[
  #include <petsc.h>
  ],$_sci_inc /usr/include/petsc,petsc petscvec petscmat petscdm petsccontrib petscsnes petscts petscsles petscfortran,$_sci_lib,$LAPACK_LIB_DIR_FLAG $LAPACK_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $M_LIB_FLAG $INC_MPI_H $MPI_LIB_DIR_FLAG $MPI_LIB_FLAG $F_LIB_FLAG,,optional)

  if test "$HAVE_PETSC" = "yes" ; then
    DEF_PETSC="#define HAVE_PETSC 1"
  fi
    
  # ---- now test for uni-petsc ---
  _sci_lib=""
  _sci_inc=""

  if test "$with_unipetsc"; then
    _sci_lib="$with_unipetsc/lib/$PETSC_SUB/$PETSC_ARCH"
    _sci_inc="$with_unipetsc/bmake/$PETSC_ARCH $with_unipetsc/include"
  fi

  SCI_TRY_LINK(PETSC_UNI,optional petsc-uni libraries and headers,[
  #include <petsc.h>
  ],$_sci_inc /usr/include/petsc,petsc petscvec petscmat petscdm petsccontrib petscsnes petscts petscsles petscfortran,$_sci_lib,$LAPACK_LIB_DIR_FLAG $LAPACK_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $M_LIB_FLAG $INC_MPIUNI_H $MPIUNI_LIB_DIR_FLAG $MPIUNI_LIB_FLAG $F_LIB_FLAG,,optional)
  if test "$HAVE_PETSC_UNI" = "yes" ; then
    DEF_PETSC_UNI="#define PETSC_UNI 1"
#    PETSC_UNI_LIB_FLAG="$PETSC_UNI_LIB_FLAG"
#    PETSC_UNI_LIB_DIR_FLAG="$PETSC_UNI_LIB_DIR_FLAG"
  fi
   
else
  AC_MSG_WARN(One or more of MPI or BLAS or LAPACK is missing.)
  AC_MSG_WARN(HAVE_MPI: $HAVE_MPI)
  AC_MSG_WARN(HAVE_MPIUNI: $HAVE_MPIUNI)
  AC_MSG_WARN(HAVE_BLAS: $HAVE_BLAS)
  AC_MSG_WARN(HAVE_LAPACK: $HAVE_LAPACK)
  AC_MSG_WARN(Not configuring PETSc.)
fi

if test "$with_petsc" && test ! "$DEF_PETSC"; then
  echo
  AC_MSG_ERROR(One or more of the PETSC components is missing.)
  echo
fi

AC_SUBST(HAVE_PETSC)
AC_SUBST(PETSC_LIB_FLAG)
AC_SUBST(PETSC_LIB_DIR_FLAG)
AC_SUBST(DEF_PETSC)
AC_SUBST(INC_PETSC_H)
AC_SUBST(PETSC_UNI_LIB_FLAG)
AC_SUBST(PETSC_UNI_LIB_DIR_FLAG)
AC_SUBST(DEF_PETSC_UNI)
AC_SUBST(INC_PETSC_UNI_H)

##  ----------------------------------------------------------------------
##  ----------  optional checks for libraries and headers  ---------------
##  ----------------------------------------------------------------------

echo
AC_MSG_NOTICE(checking for optional components......)
echo


##  --  search for Insight  ---------------------------------------------------

if test "$sci_check_insight" = "yes"; then

   # Insight is a large distribution. 
   # we do not want to check all of it. We thus check only if the
   # include and lib Insight dirs exist.

   AC_MSG_CHECKING(for Insight.)

   if test "$with_insight"; then
      dir="$with_insight"
   else
      dir="/usr/local/lib/InsightToolkit"
   fi

   if test -a "$dir/ITKConfig.cmake"; then
	INC_INSIGHT_H="`cat $dir/ITKConfig.cmake | grep 'SET(ITK_INCLUDE_DIRS' | sed 's/SET(ITK_INCLUDE_DIRS \"/;/' | sed 's/\")//' | sed 's/;/ -I/g'`" 
        INSIGHT_LIB_DIR_FLAG="`cat $dir/ITKConfig.cmake | grep 'SET(ITK_LIBRARY_DIRS' | sed 's/SET(ITK_LIBRARY_DIRS "/;/' | sed 's/")//' | sed 's/;/ -L/g'`"
	INSIGHT_LIB_FLAG="-lITKFEM -lITKIO -lITKMetaIO -lITKAlgorithms -lITKStatistics -lITKBasicFilters -lITKCommon  -litkvnl -litkvnl_algo -litknetlib -litksys -lz"
#	INSIGHT_LIB_FLAG="`cat $dir/ITKConfig.cmake | grep 'SET(ITK_LIBRARIES' | sed 's/SET(ITK_LIBRARIES//' | sed 's/)//' | sed 's/ / -l/g'`" 

      DEF_INSIGHT="#define HAVE_INSIGHT"
      HAVE_INSIGHT="yes"

      AC_MSG_RESULT($dir)
   else
      AC_MSG_RESULT(not found.)
   fi
fi

AC_SUBST(HAVE_INSIGHT)
AC_SUBST(INSIGHT_LIB_FLAG)
AC_SUBST(INSIGHT_LIB_DIR_FLAG)
AC_SUBST(INC_INSIGHT_H)
AC_SUBST(DEF_INSIGHT)

## -- search for java xalan --

## -- first, if any of the following packages are requested then we need xalan-j
if test "$pkg_Insight" = "yes"; then
	sci_check_xalan="yes"
fi

## -- second, if xalan-j is needed then check that the three .jar files are there.

if test "$sci_check_xalan" ; then
  AC_MSG_CHECKING(for xalan-j.)

  if test "$with_xalan"; then
	dir="$with_xalan"
  else
	dir="$sci_thirdparty_dir"
  fi
  if test -a $dir/bin/xalan.jar -a $dir/bin/xml-apis.jar -a $dir/bin/xercesImpl.jar; then
	AC_MSG_RESULT($dir)
	XALAN_PATH="$dir/bin/xalan.jar:$dir/bin/xml-apis.jar:$dir/bin/xercesImpl.jar"
  else	
	AC_MSG_RESULT(not found)
  fi
fi

AC_SUBST(XALAN_PATH)

##  --  if building uintah, search for TAU makefile  ---------------------

if test "$sci_check_tau" = "yes" && test "$with_tau"; then
  if test ! -f "$with_tau" ; then
    SCI_MSG_ERROR($with_tau does not exist.  Bye!)
  fi

  AC_MSG_RESULT(Using TAU makefile: $with_tau)

  cat > conftest-tau.cc <<EOF

  #include <Profile/Profiler.h>

  int main(int argc, char **argv)
  {
    TAU_PROFILE("main", "int (int, char **)", TAU_DEFAULT);
    TAU_PROFILE_INIT(argc, argv);
    return 0;
  }
EOF

  cat > conftest-tau-makefile <<EOF
include $(with_tau)
all:
  $(TAU_CXX) -o conftest-tau-a.out $(TAU_DEFS) $(TAU_INC_LUDE) conftest-tau.cc $(TAU_SHLIBS) $(TAU_MPI_LIBS)
EOF

  ac_try="make TAU_MAKEFILE=${tau} -f conftest-tau-makefile > conftestout1) 2> conftesterr1"
  (eval $ac_try) 2>&5
  ac_status=$?
  if $ac_status; then
    rm -f conftest-tau.cc conftest-tau-makefile conftest-tau-a.out conftestout1 conftesterr1
    SCI_MSG_ERROR(TAU libraries not working. Bye!)
  fi
  rm -f conftest-tau.cc conftest-tau-makefile conftest-tau-a.out conftestout1 conftesterr1

  TAU_MAKEFILE="$with_tau"
fi

AC_SUBST(TAU_MAKEFILE)

##  --  search for Babel  ----------------------------------------------------
if test "$sci_check_babel" = "yes"; then
  if test "$with_babel"; then
    _sci_lib="$with_babel/lib"
    _sci_inc="$with_babel/include"
  else
    _sci_lib="$sci_thirdparty_libs"
    _sci_inc="$sci_thirdparty_includes"
  fi
  
  SCI_TRY_LINK(SIDL,optional sidl library and headers,[
     #include <SIDL.h>
     ],$_sci_inc,sidl,$_sci_lib,,,optional)
  
  BABEL_BIN=""
  AC_CHECKING(for babel in $LIB_DIR_SIDL/../bin)
  AC_CHECK_PROG(BABEL_BIN,babel,$LIB_DIR_SIDL/../bin/babel,NONE,$LIB_DIR_SIDL/../bin)
  
  if test "$HAVE_SIDL" = "yes" && test $BABEL_BIN != "NONE"; then
     DEF_BABEL="#define HAVE_BABEL 1"
     HAVE_BABEL="yes"
  fi
  
  if test "$with_babel" && test ! "$DEF_BABEL"; then
    echo
    AC_MSG_ERROR(One or more of the Babel components is missing.)
    echo
  fi
fi # end sci_check_babel
AC_SUBST(HAVE_BABEL)
AC_SUBST(SIDL_LIB_FLAG)
AC_SUBST(SIDL_LIB_DIR_FLAG)
AC_SUBST(INC_SIDL_H)
AC_SUBST(DEF_BABEL)
AC_SUBST(BABEL_BIN)

##  --  search for tiff and jpeg -----------------------------------------

if test "$sci_check_jpeg" = "yes"; then

   if test "$with_jpeg"; then
     _sci_lib="$with_jpeg/lib"
   else
     _sci_lib="$sci_thirdparty_libs"
   fi 

   SCI_TRY_LINK(JPEG,jpeg library,,,jpeg,$_sci_lib,,,not-optional)

fi #end sci_check_jpeg

AC_SUBST(JPEG_LIB_FLAG)
AC_SUBST(JPEG_LIB_DIR_FLAG)


if test "$sci_check_tiff" = "yes"; then
   if test "$with_tiff"; then
     _sci_lib="$with_tiff/lib"
   else
     _sci_lib="$sci_thirdparty_libs"
   fi

   SCI_TRY_LINK(TIFF,tiff library,,,tiff,$_sci_lib,$M_LIB_DIR_FLAG $M_LIB_FLAG,,not-optional)

fi # end sci_check_tiff

AC_SUBST(TIFF_LIB_FLAG)
AC_SUBST(TIFF_LIB_DIR_FLAG)

##  --  search for netsolve ----------------------------------------------

if test "$sci_check_netsolve" = "yes"; then

   if test "$with_netsolve"; then
     _sci_lib="$with_netsolve"
   else
     _sci_lib="$sci_thirdparty_libs"
   fi

   SCI_TRY_LINK(NETSOLVE,netsolve library,,,netsolve,$_sci_lib,,,not-optional)
fi

AC_SUBST(NETSOLVE_LIB_FLAG)
AC_SUBST(NETSOLVE_LIB_DIR_FLAG)


 ##  --  search for glut, glui and oogl -----------------------------------

if test "$sci_check_glut" = "yes"; then

   libs="$GL_LIB_DIR_FLAG $GL_LIB_FLAG $X_LIB_FLAG $GLU_OTHERS"

   if test "$with_glut"; then
     _sci_lib="$with_glut/$BASE_LIB_DIR"
     _sci_inc="$with_glut/include"
   else
     _sci_lib="$sci_thirdparty_libs"
     _sci_inc="$sci_thirdparty_includes"
   fi

   SCI_TRY_LINK(GLUT,glut library and headers,[
   #include <GL/glut.h>
   ],$_sci_inc,glut,$_sci_lib,$libs,,not-optional)
fi #end sci_check_glut

AC_SUBST(INC_GLUT_H)
AC_SUBST(GLUT_LIB_FLAG)
AC_SUBST(GLUT_LIB_DIR_FLAG)

if test "$sci_check_glui" = "yes"; then
   if test "$with_glui"; then
     _sci_lib="$with_glui/$BASE_LIB_DIR"
     _sci_inc="$with_glui/include"
   else
     _sci_lib="$sci_thirdparty_libs"
     _sci_inc="$sci_thirdparty_includes"
   fi

   SCI_TRY_LINK(GLUI,glui library and headers,[
   #include <GL/glut.h>
   #include <glui.h>
   ],$_sci_inc,glui,$_sci_lib,$libs $GLUT_LIB_DIR_FLAG $GLUT_LIB_FLAG,,not-optional)

fi # end sci_check_glui
AC_SUBST(INC_GLUI_H)
AC_SUBST(GLUI_LIB_FLAG)
AC_SUBST(GLUI_LIB_DIR_FLAG)


if test "$sci_check_oogl" = "yes"; then
  # Modify _sci_dirs to have correct path information:
  if test "$with_oogl"; then
    _sci_lib="$with_oogl/lib"
    _sci_inc="$with_oogl/include"
  else
    for i in $sci_thirdparty_dir; do
      if test ! -d "$i"; then
        continue
      fi
  
      #this is where the lib actually lives
      if test ! -d "$i/OOGL/lib"; then
        continue
      fi
  
      _sci_lib="$_sci_dirs $i/OOGL/lib"
      _sci_inc="$_sci_dirs $i/OOGL/include"
    done
  fi
  # Only check for OOGL if the user did not say without-oogl.
  if test "$with_oogl" != "no"; then
    SCI_TRY_LINK(OOGL,oogl library and headers,[
      #include <oogl/basicTexture.h>
      #include <oogl/shadedPrim.h>
      ],$_sci_inc,oogl,$_sci_lib,$INC_GLUT_H $INC_GLUI_H,,optional)
  fi
fi # end sci_check_oogl

if test "$HAVE_OOGL" = "yes"; then
  DEF_OOGL="#define HAVE_OOGL 1"
fi

AC_SUBST(INC_OOGL_H)
AC_SUBST(DEF_OOGL)
AC_SUBST(OOGL_LIB_FLAG)
AC_SUBST(OOGL_LIB_DIR_FLAG)

if test "$sci_check_audio" = "yes"; then
  _sci_lib="$with_audio"

  SCI_TRY_LINK(AUDIO,audio library,,,audio audiofile,$_sci_lib,,,optional)

  if test "$HAVE_AUDIO" = "yes"; then
    DEF_AUDIO="#define HAVE_AUDIO 1"
  fi

  if test "$with_audio" && test ! "$DEF_AUDIO"; then
    echo
    AC_MSG_ERROR(One or more of the AUDIO components is missing.)
    echo
  fi

fi # end sci_check_audio

AC_SUBST(DEF_AUDIO)
AC_SUBST(HAVE_AUDIO)
AC_SUBST(AUDIO_LIB_FLAG)
AC_SUBST(AUDIO_LIB_DIR_FLAG)

##  --  search for teem  -------------------------------------------------

if test "$sci_check_teem" = "yes"; then
  if test "$with_teem"; then
    _sci_lib="$with_teem/lib"
    _sci_inc="$with_teem/include"
  else
    _sci_lib="$sci_thirdparty_libs"
    _sci_inc="$sci_thirdparty_includes"
  fi
  
  SCI_TRY_LINK(TEEM,teem library and headers,[
   #include <teem/nrrd.h>
   ],$_sci_inc,teem,$_sci_lib,-lm,,not-optional)

  if test "$HAVE_TEEM" = "yes"; then
     DEF_TEEM="#define HAVE_TEEM 1"
  fi  
fi
AC_SUBST(TEEM_LIB_FLAG)
AC_SUBST(TEEM_LIB_DIR_FLAG)
AC_SUBST(INC_TEEM_H)
AC_SUBST(DEF_TEEM)

##  --  search for mdsplus  -------------------------------------------------

if test "$sci_check_mdsplus" = "yes"; then
  if test "$with_mdsplus"; then
    _sci_lib="$with_mdsplus/lib"
    _sci_inc="$with_mdsplus/include"
  else
    _sci_lib="/usr/local/mdsplus/lib"
    _sci_inc="/usr/local/mdsplus/include"
  fi

# mdsplus is a C lib so it has to be checked as a C file not CC..
  cxx="$CXX"
  CXX="$CC"

  ext="$ac_ext"
  ac_ext="c"

  SCI_TRY_LINK(MDSPLUS,mdslib library and headers,[
   #include <mdslib.h>
   ],$_sci_inc,MdsLib_client,$_sci_lib,,,not-optional)

  if test "$HAVE_MDSPLUS" = "yes"; then
     DEF_MDSPLUS="#define HAVE_MDSPLUS 1"
  fi
  
  if test "$with_mdsplus" && test ! "$DEF_MDSPLUS"; then
    echo
    AC_MSG_ERROR(One or more of the MDSPLUS components is missing.)
    echo
  fi

# Restore the previous settings.
  CXX="$cxx"
  ac_ext="$ext"
fi #end sci_check_mdsplus

AC_SUBST(MDSPLUS_LIB_FLAG)
AC_SUBST(MDSPLUS_LIB_DIR_FLAG)
AC_SUBST(INC_MDSPLUS_H)
AC_SUBST(DEF_MDSPLUS)

##  --  search for hdf5  -------------------------------------------------

if test "$sci_check_hdf5" = "yes"; then
  if test "$with_hdf5"; then
    _sci_lib="$with_hdf5/lib"
    _sci_inc="$with_hdf5/include"
  else
    _sci_lib="/usr/local/lib"
    _sci_inc="/usr/local/include"
  fi

  SCI_TRY_LINK(HDF5,hdf5 library and headers,[
   #include <hdf5.h>
   ],$_sci_inc,hdf5,$_sci_lib,,,optional)

  if test "$HAVE_HDF5" = "yes"; then
     DEF_HDF5="#define HAVE_HDF5 1"
  fi
  
  if test "$with_hdf5" && test ! "$DEF_HDF5"; then
    echo
    AC_MSG_ERROR(One or more of the HDF5 components is missing.)
    echo
  fi
fi #end sci_check_hdf5

AC_SUBST(HDF5_LIB_FLAG)
AC_SUBST(HDF5_LIB_DIR_FLAG)
AC_SUBST(INC_HDF5_H)
AC_SUBST(DEF_HDF5)

##  --  search for perfex  -----------------------------------------------

if test "$sci_check_perfex" = "yes"; then
  _sci_lib="$sci_thirdparty_libs"

  SCI_TRY_LINK(PERFEX,perfex library,,,perfex,$_sci_lib,,,optional)

  if test "$HAVE_PERFEX" = "yes"; then
    DEF_PERFEX="#define HAVE_PERFEX 1"
  else
    AC_MSG_ERROR(One or more of the PERFEX components is missing.)
  fi
fi
AC_SUBST(PERFEX_LIB_FLAG)
AC_SUBST(PERFEX_LIB_DIR_FLAG)
AC_SUBST(DEF_PERFEX)


##  --  search for crypto --------------------------------

if test "$sci_check_crypto" = "yes"; then
  if test "$with_crypto"; then
    _sci_lib="$with_crypto/lib"
  else
    _sci_lib="$with_globus/lib"
  fi
  
  SCI_TRY_LINK(CRYPTO,crypto library,,,crypto,$_sci_lib,,,optional) 
  
  if test "$HAVE_CRYPTO" = "yes"; then
    DEF_CRYPTO="#define HAVE_CRYPTO 1"
  fi
  
  if test "$with_crypto" && test ! "$DEF_CRYPTO"; then
    echo
    AC_MSG_ERROR(One or more of the CRYPTO components is missing.)
    echo
  fi
fi # end sci_check_crypto

AC_SUBST(CRYPTO_LIB_FLAG)
AC_SUBST(CRYPTO_LIB_DIR_FLAG)
AC_SUBST(DEF_CRYPTO)

##  --  search for ssl ---------------------------------

if test "$sci_check_ssl" = "yes"; then
  if test "$with_ssl"; then
    _sci_lib="$with_ssl/lib"
  else
    _sci_lib="$sci_thirdparty_libs"
  fi

  SCI_TRY_LINK(SSL,ssl library,,,ssl crypto,$_sci_lib,,,optional)  
  
  if test "$HAVE_SSL" = "yes"; then
    DEF_SSL="#define HAVE_SSL 1"
  fi
  
  if test "$with_ssl" && test ! "$DEF_SSL"; then
    echo
    AC_MSG_ERROR(One or more of the SSL components is missing.)
    echo
  fi
fi # end sci_check_ssl

AC_SUBST(SSL_LIB_FLAG)
AC_SUBST(SSL_LIB_DIR_FLAG)
AC_SUBST(DEF_SSL)

##  --  globus nexus  ----------------------------------------------------
if test "$sci_check_globus" = "yes"; then

  if test "$with_globus"; then
    _sci_lib="$with_globus/lib"
    _sci_inc="$with_globus/include"
  else
    _sci_lib="$sci_thirdparty_libs"
    _sci_inc="$sci_thirdparty_includes"
  fi
  
  if test ! "$DEF_SSL"; then
    AC_MSG_ERROR(SSL library not found.  This will cause globus check to fail.  Use --with-ssl=path_to_ssl.)
  fi
  if test ! "$DEF_CRYPTO"; then
    AC_MSG_ERROR(CRYPTO library not found.  This will cause globus check to fail.  Use --with-crypto=path_to_crypto)
  fi

  MISC_LIB_FLAG="$SSL_LIB_DIR_FLAG $SSL_LIB_FLAG $CRYPTO_LIB_DIR_FLAG $CRYPTO_LIB_FLAG"
  
  SCI_TRY_LINK(GLOBUS,globus library and headers,,
  $_sci_inc, globus_dc pthread,$_sci_lib,$MISC_LIB_FLAG,,optional)

  # If no with-globus on command line, but we have found globus, try
  # the one we found when doing the following AC_CHECK_FILES:
  if test -z "$with_globus"; then
    with_globus_lib=$LIB_DIR_GLOBUS/..
  else
    with_globus_lib=$with_globus
  fi

  # These globus libs need SCIRun thread files to link.  Because we
  # don't have them yet (we haven't compiled SCIRun yet...), we
  # can't test for linking.  Therefore we just test for existence.
  AC_CHECK_FILES($with_globus_lib/lib/libglobus_common.so $with_globus_lib/lib/libglobus_nexus.so $with_globus_lib/lib/libglobus_io.so $with_globus_lib/lib/libglobus_utp.so $with_globus_lib/lib/libglobus_mp.so $with_globus_lib/lib/libglobus_gss_assist.so $with_globus_lib/lib/libglobus_gss.so $with_globus_lib/lib/libglobus_gaa.so $with_globus_lib/include/globus_nexus.h,,HAVE_GLOBUS="")

  if test "$HAVE_GLOBUS" = "yes"; then
    DEF_GLOBUS="#define HAVE_GLOBUS 1"
    GLOBUS_LIB_FLAG="$GLOBUS_LIB_FLAG -lglobus_common -lglobus_nexus -lglobus_io -lglobus_mp -lglobus_gss_assist -lglobus_gss -lglobus_gaa $MISC_LIB_FLAG"
    INC_GLOBUS_H=-I"$with_globus"/include
  fi
  
  if test "$with_globus" && test ! "$DEF_GLOBUS"; then
    echo
    AC_MSG_ERROR(One or more of the GLOBUS components is missing.)
    echo
  fi

fi #end sci_check_globus

AC_SUBST(GLOBUS_LIB_FLAG)
AC_SUBST(GLOBUS_LIB_DIR_FLAG)
AC_SUBST(INC_GLOBUS_H)
AC_SUBST(DEF_GLOBUS)
AC_SUBST(HAVE_GLOBUS)

##  --  search for Qt  ----------------------------------------------------
if test "$sci_check_qt" = "yes"; then
  if test "$with_qt"; then
    _sci_lib="$with_qt/lib"
  else
    _sci_lib="$sci_thirdparty_libs"
  fi

  _qt_inc="$with_qt/include"
  
  # Only looking for one .h file... assuming all the rest are in the same place.
  SCI_TRY_LINK(QT_MT,qt library,[
     #include <qcanvas.h>
     ],$_qt_inc,qt-mt,$_sci_lib,,,optional)  

  QT_MOC_BIN=""
  AS_MESSAGE(checking for moc in $LIB_DIR_QT_MT/../bin...)
  AC_CHECK_PROG(QT_MOC_BIN,moc,$LIB_DIR_QT_MT/../bin/moc,NOMOC,$LIB_DIR_QT_MT/../bin)
  
  if test "$HAVE_QT_MT" = "yes" && test $QT_MOC_BIN != "NOMOC"; then
     DEF_QT="#define HAVE_QT 1"
     HAVE_QT="yes"
  fi
  
  if test "$with_qt" && test ! "$DEF_QT"; then
    echo
    AC_MSG_ERROR(One or more of the QT components is missing.)
    echo
  fi
fi # end sci_check_qt

AC_SUBST(HAVE_QT)
AC_SUBST(QT_MT_LIB_FLAG)
AC_SUBST(QT_MT_LIB_DIR_FLAG)
AC_SUBST(QT_MOC_BIN)
AC_SUBST(INC_QT_MT_H)
AC_SUBST(DEF_QT)

##  --  search for VDT  ----------------------------------------------------

if test "$sci_check_vdt" = "yes"; then
  if test "$with_vdt"; then
    _vdtbase="$with_vdt"
  else
    _vdtbase=$sci_thirdparty_dir
  fi

  for i in $_vdtbase; do
    if test ! -d "$i"; then
      continue
    fi
  
    #from here we should find src/vdt/bin
    if test ! -d "$i/src/vdt/bin"; then
      continue
    fi
  
    guess=`sh $i/src/vdt/bin/config.guess`
    ad=`sh $i/src/vdt/bin/config.sub $guess`
  
    #this is where the lib actually lives
    if test ! -d "$i/src/vdt/VDT/lib/$ad"; then
      continue
    fi
    _vdt_inc="$i/src/vdt/VDT/src $i/src/vdt/G3D/include $i/src/vdt/Ckit/include $i/src/vdt/LOBS/include $i/src/vdt/PiRaT/include $i/src/vdt/Meschach $i/src/vdt/lp_solve_2.2 $i/src/vdt/Elixir/include"
    _vdt_lib_path="$i/src/vdt/VDT/lib/$ad $i/src/vdt/Elixir/lib/$ad $i/src/vdt/lp_solve_2.2/$ad $i/src/vdt/PiRaT/lib/$ad $i/src/vdt/LOBS/lib/$ad $i/src/vdt/G3D/lib/$ad $i/src/vdt/Ckit/lib/$ad $i/src/vdt/Meschach/$ad"

    _vdt_libs="esi elixir lpk pirat lobs3d lobs2d lobs1d g3d ckit meschach vdt"

    MISC_LIB_FLAGS="$X_LIB_DIR_FLAG $X_LIB_FLAG $M_LIB_FLAG $LEX_LIB_FLAG"

    SCI_TRY_LINK(VDT,vdt libraries and headers,[
    #include <vdt.h>
    ],$_vdt_inc,$_vdt_libs,$_vdt_lib_path,$MISC_LIB_FLAGS,[
    VDT_triangulate();
    ],not-optional) 
 
    if test "$HAVE_VDT" = "yes"; then
      DEF_VDT="#define HAVE_VDT 1"
      break
    fi
  done

fi # end sci_check_vdt

AC_SUBST(DEF_VDT)
AC_SUBST(VDT_LIB_FLAG)
AC_SUBST(VDT_LIB_DIR_FLAG)
AC_SUBST(INC_VDT_H)


##  --  search for hypre stuff  ------------------------------------------

if test "$sci_check_hypre" = "yes"; then

  if test "$with_hypre"; then
    _sci_lib="$with_hypre/lib $with_hypre/src/hypre/lib"
    _sci_inc="$with_hypre/include $with_hypre/src/hypre/include"
  else
    _sci_lib="$sci_thirdparty_libs"
    _sci_inc="$sci_thirdparty_includes"
  fi

  SCI_TRY_LINK(HYPRE,HYPRE libraries and headers,[
    #include <HYPRE.h>
    ],$_sci_inc,HYPRE_struct_mv HYPRE_struct_ls HYPRE_blas HYPRE_LSI,
    $_sci_lib,,,optional)

  if test "$HAVE_HYPRE" = "yes" ; then
    DEF_HYPRE="#define HAVE_HYPRE 1"
  fi
fi

if test "$with_hypre" && test ! "$DEF_HYPRE"; then
  echo
  AC_MSG_ERROR(One or more of the HYPRE components is missing.)
  echo
fi

AC_SUBST(HAVE_HYPRE)
AC_SUBST(DEF_HYPRE)
AC_SUBST(HYPRE_LIB_FLAG)
AC_SUBST(HYPRE_LIB_DIR_FLAG)
AC_SUBST(INC_HYPRE_H)


##  --  finis flags  ------------------------------------------------
case $host in
  *-irix*)
    LDFLAGS="$LDFLAGS -J\$(MAKE_PARALLELISM) $(TLINK)"
    ;;
  *linux*)
    ;;
  *aix*)
    ;;
  *dec*)
    ;;
  *)
    ;;
esac


## -- grep of the code suggests only those dependent upon CCA need this.
##  --  search for uuid functions, either uuid_create or uuid_generate ----

if test "$sci_check_uuid" = "yes"; then
  # This is typical on SGI and ??
  AC_MSG_CHECKING(for uuid_create in sys/uuid.h)
  AC_TRY_COMPILE([#include <sys/types.h>
               extern "C" {
                  #include <sys/uuid.h>
               }],
  	[char* uuid_str;
  	 uuid_t uuid;
  	 uint_t status;
  	 uuid_create(&uuid, &status);],
  	[have_uuid=yes
           AC_DEFINE(HAVE_SYS_UUID_H)], have_uuid=no)
  AC_MSG_RESULT($have_uuid)
  
  if test "$have_uuid" != "yes"; then
    # This is typical on linux and ??

    AC_MSG_CHECKING(for uuid_generate in uuid/uuid.h)
    AC_TRY_COMPILE([
  extern "C" { // The linux uuid.h doesn't have this, so we need exter C here
  #include <uuid/uuid.h>
  }],
  	[uuid_t uuid;
  	 char uuid_str[64];
  	 uuid_generate(uuid);
  	 uuid_unparse(uuid, uuid_str);],
  	[have_uuid=yes
           AC_DEFINE(HAVE_UUID_UUID_H)], have_uuid=no)
    AC_MSG_RESULT($have_uuid)
  fi
  
  if test "$have_uuid" = "yes"; then
    AC_MSG_CHECKING(for uuid library)
    old_LIBS="$LIBS"
    for i in -luuid ""; do
      LIBS="$old_LIBS $i"
      AC_TRY_LINK([
  #if HAVE_SYS_UUID_H
  #include <sys/types.h>
  extern "C" { // SGI uuid.h doesn't have this, so we need extern C here
  #include <sys/uuid.h>
  }
  #define UUID_CREATE
  #else
  #if HAVE_UUID_UUID_H
  extern "C" { // The linux uuid.h does not have this, so we need extern C here
  #include <uuid/uuid.h>
  }
  #define UUID_GENERATE
  #else
  #error We need either sys/uuid.h or uuid/uuid.h
  #endif
  #endif
  	], [
    uuid_t uuid;
  #if defined(UUID_CREATE)
    char* uuid_str;
    uint_t status;
    uuid_create(&uuid, &status);
    uuid_to_string(&uuid, &uuid_str, &status);
  #else
  #if defined( UUID_GENERATE)
    char uuid_str[64];
    uuid_generate( uuid );
    uuid_unparse(uuid, uuid_str);
  #endif
  #endif
  	], [
    UUID_LIB_FLAG=$i
    break
  	])
    done
    if test "$UUID_LIB_FLAG" != unknown; then
      if test -z "$UUID_LIB_FLAG"; then
        AC_MSG_RESULT(none required)
      else
        AC_MSG_RESULT($UUID_LIB_FLAG)
      fi
    else
      AC_MSG_RESULT(not found)
      have_uuid=no
    fi
    LIBS="$old_LIBS"
  fi

fi # end sci_check_uuid

AC_SUBST(UUID_LIB_FLAG)

##  --  See if we have everything needed for SCIRun2 ----------------------

if test "$enable_scirun2" = "yes"; then
  ## For now, don't check for uuid if on macintosh.
  if test "$have_uuid" != "yes"; then
    case $host in
      *linux*)
        SCI_MSG_ERROR(uuid_generate function and libuuid not found, must install ext2fs development tools)
        ;;
      *)
        SCI_MSG_ERROR(Cannot find uuid_generate or uuid_create, required for SCIRun2)
        ;;
    esac
    exit 1
  fi
  DEF_SCIRUN2="#define SCIRUN2"
  BUILD_SCIRUN2="yes"
else
  DEF_SCIRUN2=""
  BUILD_SCIRUN2="no"
fi
AC_SUBST(DEF_SCIRUN2)
AC_SUBST(BUILD_SCIRUN2)

##  ---------------------------------------------------------------------
##  --------------------  generate output files  ------------------------
##  ---------------------------------------------------------------------
SCI_ARG_VAR(CXXFLAGS,C++ compiler flags)
abs_conftop_dir=`pwd`;
AC_SUBST(abs_conftop_dir)
abs_srcdir=`(cd ${srcdir};pwd)`
AC_SUBST(abs_srcdir)

AC_CONFIG_FILES(configVars.mk Makefile on-the-fly-libs/Makefile 
                disjointPackageMakefile sci_testdefs.h)

##   Only do the following if configuring with Uintah
##   Add a list of Uintah specific output files.
if test "$pkg_uintah" = "yes" ; then
   AC_CONFIG_FILES(Packages/Uintah/tools/fspec.pl)
fi 

#  ignore --no-create and --no-recursion flags
no_create=no
no_recursion=no

# these are to alias pwd to the right thing in the config.status
# script so that pwd returns the same value for a path even if you
# get there by two different paths (symbolic links)
cat > sedscript << SCIEOF
/debug=false/i\\
alias pwd=$sci_pwd
SCIEOF

AC_CONFIG_COMMANDS_POST(sed -f sedscript < config.status > config_test.status;\
			    mv config_test.status config.status ;\
			    chmod 755 config.status)


AC_CONFIG_COMMANDS([do_stamp],[echo timestamp > stamp-h ])
AC_OUTPUT

##  check whether sci_defs.h has changed (due to a re-configure)
if cmp -s sci_defs.h sci_testdefs.h 2>/dev/null; then
  AC_MSG_RESULT(sci_defs.h is unchanged)
  rm -f sci_testdefs.h
else
  mv sci_testdefs.h sci_defs.h
  AC_MSG_WARN(sci_defs.h has changed.)
fi

#### Clean up: ####
## Possible temporary template directories
rm -rf ii_files
## sedscript
rm -f sedscript
