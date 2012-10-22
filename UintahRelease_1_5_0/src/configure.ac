##
## The MIT License
##
## Copyright (c) 1997-2012 The University of Utah
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.
##

##  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##
##  This configure.ac file has some specialized sections for doing certain
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
##  FOR AN EXAMPLE OF ADDING A NEW LIBRARY LOOK AT THE HDF5 LIBRARY
##  IT IS DOCUMENTED AND CAN BE USED AS A COPY PASTE EXAMPLE.
##  IF YOU NEED MORE HELP SEE DAV, MARTY, ALLEN, OR YARDEN.
##
##  PLEASE DO NOT COMMIT ANY CHANGES WITH OUT HAVING ONE OF THE 
##  ABOVE REVIEW YOUR CHANGES.
##
##  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##  --------------------------------------------------------------------
##  ------------------------  initialize  ------------------------------
##  --------------------------------------------------------------------

##  initialize configure 
AC_INIT([Uintah],[1.4],[uintah-developer@gforge.sci.utah.edu])

##  require autoconf version 2.68
##     The reason to require the latest autoconf is to try and get everyone who
##     updates configure to use the same version.  By doing so, the SVN diffs
##     don't look completely different from minor changes.  (This happens
##     if people switch back and forth between different versions of autoconf.)
##
##     type "autoconf --version" to find out what version you have.
##
AC_PREREQ(2.68)

AC_COPYRIGHT(
##
## The MIT License
##
## Copyright (c) 1997-2012 The University of Utah
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.
##
)

##
## Any error/warning/note/remark messages that need to be turned off
## globally should be appended to this variable.  This should only be
## done if the message is pervasive and has been determined to be
## benign.  WOFF_NUMBERS should be a 'space' delimited list of numbers.
## Eg: '123 124 155'
##
WOFF_NUMBERS=

##
## Used to store warnings that should be redisplayed to the user at the
## end of configure in hopes that the user actually sees them. 
##
## Use 'add_important_warning()' to add a warning.
##
declare -a IMPORTANT_WARNINGS
NUM_IMPORTANT_WARNINGS=0
add_important_warning()
{
   # Note: need '[[' below to get autoconf to create a single '[' in the actual configure file.
   IMPORTANT_WARNINGS[[$NUM_IMPORTANT_WARNINGS]]=$1
   ((NUM_IMPORTANT_WARNINGS++))
}

## Clean up files that could be left around if configure was 
## interrupted (by control-c, ^c) because I do not know how to add
## cleanup to the builtin AC function that uses the 'trap' command.
rm -rf ii_files sedscript

AC_CONFIG_SRCDIR(configVars.mk.in)

##  set the location of config.guess, config.sub, and install.sh
AC_CONFIG_AUX_DIR(build_scripts)

## If user tried to configure in the src directory, this will block it.
if test -f Makefile.in; then
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
SCI_ARG_VAR(NVCC, CUDA compiler driver command)
SCI_ARG_VAR(NVCC_CFLAGS, CUDA C compiler flags)
SCI_ARG_VAR(NVCC_CXXFLAGS, CUDA C++ compiler flags)
SCI_ARG_VAR(NVCC_LDFLAGS, CUDA linker flags)

AC_CANONICAL_HOST

## find the machine's pwd, and don't use the sh built-in pwd.
## sh pwd can print the wd with symbolic links depending on
## how you got to that directory, and thus has the potential
## to print different pwd's for the same wd, and change files
## like sci_defs.h

sci_pwd=""
for my_pwd in pwd /bin/pwd /usr/bin/pwd; do

  if ! test -x $my_pwd; then
      # This pwd isn't an executable (or doesn't exist)... skip it.
      continue
  fi

  mkdir -p pwd_test
  ln -s pwd_test pwd_temp

  cd pwd_test
  pwd_1=`$my_pwd`

  cd ../pwd_temp
  pwd_2=`$my_pwd`

  cd ..
  rm -fr pwd_test pwd_temp
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
##  ------------------  optional variable intialization  ---------------
##  --------------------------------------------------------------------

# Setting variables to 'no' isn't required, but is useful in displaying
# some error condition information to the user.
HAVE_HDF5="no"
HAVE_BOOST="no"
HAVE_TABPROPS="no"
HAVE_SPATIALOPS="no"
HAVE_EXPRLIB="no"

##  --------------------------------------------------------------------
##  ------------------  check for all arguments     --------------------
##  --------------------------------------------------------------------

##
##  Use the SCI_ARG_WITH and SCI_ARG_ENABLE macros to check for arguments
##


SCI_ARG_WITH(atlas,
  AC_HELP_STRING(--with-atlas=DIR,
                 Use the atlas (for lapack/blas) installation in DIR),
  [with_atlas="$withval"
   sci_required_atlas=yes],
  [with_atlas=""])
SCI_ARG_WITH(mkl,
  AC_HELP_STRING(--with-mkl=DIR,
                 Use the mkl (for lapack/blas) installation in DIR),
  [with_mkl="$withval"
   sci_required_mkl=yes],
  [with_mkl=""])
SCI_ARG_WITH(acml,
  AC_HELP_STRING(--with-acml=DIR,
                 Use the acml (for lapack/blas) installation in DIR),
  [with_acml="$withval"
   sci_required_acml=yes],
  [with_acml=""])
SCI_ARG_WITH(blas,
  AC_HELP_STRING(--with-blas=DIR,
                 Use the blas installation in DIR),
  [with_blas="$withval"
   sci_required_blas=yes],
  [with_blas=""])
SCI_ARG_WITH(electric_fence,
  AC_HELP_STRING(--with-electric-fence=DIR,
                 Use Electric Fence (Memory Tracking) library in DIR),
  [with_electric_fence="$withval"
   sci_required_electric_fence=yes],
  [with_electric_fence=""])
SCI_ARG_WITH(lapack,
  AC_HELP_STRING(--with-lapack=DIR, 
                 Use the lapack installation in DIR),
  [with_lapack="$withval"
   sci_required_lapack=yes],
  [with_lapack=""])
SCI_ARG_WITH(teem,
  AC_HELP_STRING(--with-teem=DIR, 
                 Use the teem installation in DIR/lib),
  [with_teem="$withval"
   sci_required_teem=yes],
  [with_teem=""])
SCI_ARG_WITH(fortran,
  AC_HELP_STRING(--with-fortran=DIR, 
                 Use the fortran (eg: g2c) installation in DIR/lib),
  [with_fortran="$withval"
   sci_required_fortran=yes],
  [with_fortran=""])
SCI_ARG_WITH(ofed,
  AC_HELP_STRING(--with-ofed=DIR,
                 Specify that OFED - Infiniband - is needed for the mpi installation),
  [with_ofed="$withval"
   sci_required_ofed=no],
  [with_ofed=""])
SCI_ARG_WITH(mx,
  AC_HELP_STRING(--with-mx=DIR,
                 Specify that mx - Myrinet - is needed for the mpi installation),
  [with_mx="$withval"
   sci_required_mx=no],
  [with_mx=""])
SCI_ARG_WITH(mpi,
  AC_HELP_STRING(--with-mpi=DIR,
                 Use the mpi installation in DIR),
  [with_mpi="$withval"
   sci_required_mpi=yes],
  [with_mpi=""])
SCI_ARG_WITH(mpi_lib,
  AC_HELP_STRING(--with-mpi-lib=DIR,
                 Use the mpi libraries installed in DIR),
  [with_mpi_lib="$withval"
   sci_required_mpi=yes],
  [with_mpi_lib=""])
SCI_ARG_WITH(mpi_include,
  AC_HELP_STRING(--with-mpi-include=DIR,
                 Use the mpi include files installed in DIR),
  [with_mpi_include="$withval"
   sci_required_mpi=yes],
  [with_mpi_include=""])
SCI_ARG_WITH(mpi_extra_libs,
  AC_HELP_STRING(--with-mpi-extra-libs=/path/to/lib1.so,
                 Use this additional library(or libraries - comma separated) for mpi linking),
  [with_mpi_extra_libs="$withval"],
  [with_mpi_extra_libs=""])
SCI_ARG_WITH(cuda,
  AC_HELP_STRING(--with-cuda=DIR,
                 Use the CUDA installation in DIR),
  [with_cuda="$withval"
   sci_required_cuda=yes],
  [with_cuda=""])  
SCI_ARG_WITH(hypre,
  AC_HELP_STRING(--with-hypre=DIR,
                 Use the hypre installation in DIR),
  [with_hypre="$withval"
   sci_required_hypre=yes],
  [with_hypre=""])
SCI_ARG_WITH(petsc,
  AC_HELP_STRING(--with-petsc=DIR,
                 Use the petsc installation in DIR),
  [with_petsc="$withval"
   sci_required_petsc=yes],
  [with_petsc=""])
SCI_ARG_WITH(boost,
  AC_HELP_STRING(--with-boost=DIR,
                 Use the Boost library in DIR),
  [with_boost="$withval"
   sci_required_boost=yes],
  [with_boost=""])
SCI_ARG_WITH(spatialops,
  AC_HELP_STRING(--with-spatialops=DIR,
                 Use the SpatialOps library in DIR),
  [with_spatialops="$withval"
   sci_required_spatialops=yes],
  [with_spatialops=""])
SCI_ARG_WITH(exprlib,
  AC_HELP_STRING(--with-exprlib=DIR,
                 Use the ExprLib library in DIR),
  [with_exprlib="$withval"
   sci_required_exprlib=yes],
  [with_exprlib=""])
SCI_ARG_WITH(tabprops,
  AC_HELP_STRING(--with-tabprops=DIR,
                 Use the TabProps library in DIR),
  [with_tabprops="$withval"
   sci_required_tabprops=yes],
  [with_tabprops=""])
SCI_ARG_WITH(radprops,
  AC_HELP_STRING(--with-radprops=DIR,
                 Use the RadProps library in DIR),
  [with_radprops="$withval"
   sci_required_radprops=yes],
  [with_radprops=""])
SCI_ARG_WITH(zoltan,
  AC_HELP_STRING(--with-zoltan=DIR,
                 Use the Zoltan installation in DIR),
  [with_zoltan="$withval"
   sci_required_zoltan=yes],
  [with_zoltan=""])
SCI_ARG_WITH(tau,
  AC_HELP_STRING(--with-tau=<tau-makefile>,
                 Specify a Tau Makefile),
  [with_tau="$withval"
   sci_required_tau=yes],
  [with_tau=""])
  
SCI_ARG_WITH(papi,
  AC_HELP_STRING(--with-papi=DIR,
                 Use the papi installation in DIR),
  [with_papi="$withval"
   sci_required_papi=yes],
  [with_papi=""])
  
SCI_ARG_WITH(gperftools,
  AC_HELP_STRING(--with-gperftools=DIR,
                 Use the Google Perftools installation in DIR),
  [with_gperftools="$withval"
   sci_required_gperftools=yes],
  [with_gperftools=""])
  
SCI_ARG_WITH(unp,
  AC_HELP_STRING(--with-unp=DIR,
                 Use the unp installation in DIR),
  [with_unp="$withval"
   sci_required_unp=yes],
  [with_unp=""])

SCI_ARG_WITH(zlib,
  AC_HELP_STRING(--with-zlib=DIR,
                 Use the zlib installation in DIR),
  [with_zlib="$withval"
   sci_required_zlib=yes],
  [with_zlib=""])
SCI_ARG_WITH(jpeg,
  AC_HELP_STRING(--with-jpeg=DIR,
                 Use the jpeg installation in DIR),
  [with_jpeg="$withval"
   sci_required_jpeg=yes],
  [with_jpeg=""])
SCI_ARG_WITH(tiff,
  AC_HELP_STRING(--with-tiff=DIR,
                 Use the tiff installation in DIR),
  [with_tiff="$withval"
   sci_required_tiff=yes],
  [with_tiff=""])

SCI_ARG_WITH(xalan,
  AC_HELP_STRING(--with-xalan=DIR,
                 Use the xalan installation in DIR),
  [with_xalan="$withval"
   sci_required_xalan=yes],
  [with_xalan=""])

SCI_ARG_WITH(libxml2,
  AC_HELP_STRING(--with-libxml2=DIR,
                 Use the libxml2 installation in DIR.  Set 'DIR' to 'built-in' if explicit linking of xml2 is NOT required.),
  [with_libxml2="$withval"
   sci_required_libxml2=yes],
  [with_libxml2=""])

SCI_ARG_WITH(hdf5,
  AC_HELP_STRING(--with-hdf5=DIR,
                 Use the hdf5 installation in DIR),
  [with_hdf5="$withval"		      # with_hdf5 is the directory var
   sci_required_hdf5=yes],		      # flag to check for the libs
  [with_hdf5=""])		      # undefinded otherwise

SCI_ARG_WITH(freetype,
  AC_HELP_STRING(--with-freetype=DIR,
                 Use the Freetype2 installation in DIR),
  [with_freetype="$withval"
   sci_required_freetype=yes],
  [with_freetype=""])
SCI_ARG_WITH(ftgl,
  AC_HELP_STRING(--with-ftgl=DIR,
                 Use the Freetype GL installation in DIR),
  [with_ftgl="$withval"
   sci_required_ftgl=yes
   sci_required_freetype=yes],
  [with_ftgl=""])
SCI_ARG_WITH(visit,
  AC_HELP_STRING(--with-visit=DIR,
                 Turn on VisIt visualization tool support.  Point to VisIt installation.),
  [with_visit="$withval"],
  [with_visit=""])

SCI_ARG_ENABLE(wasatch_3p,
  AC_HELP_STRING(--enable-wasatch_3p,
                 Download and build the Wasatch Thirdparty libraries - TabProps ExprLibs SpatialOps RadProps.),
  [wasatch_3p="yes"],
  [wasatch_3p="no"])

SCI_ARG_ENABLE(wasatch_in_arches,
  AC_HELP_STRING(--enable-wasatch-in-arches,
                 Turn on the use of Wasatch objects in the Arches component.),
  [wasatch_in_arches="yes"],
  [wasatch_in_arches="no"])

SCI_ARG_ENABLE(static,
  AC_HELP_STRING(--enable-static,
                 Build a statically linked version of Uintah.  Set to 'full' to force full static link.),
  [static="$enableval"],
  [static="no"])
SCI_ARG_ENABLE(verbosity,
  AC_HELP_STRING(--enable-verbosity,
                 Turn on verbose configure output),
  [verbosity="yes"],
  [verbosity="no"])
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
  [enable_64bit='no'])
SCI_ARG_ENABLE(32bit,
  AC_HELP_STRING(--enable-32bit,
                 Compile in 32 bit mode),
  [enable_32bit="$enableval"],
  [enable_32bit='no'])
SCI_ARG_ENABLE(output-avg-walltime,
  AC_HELP_STRING(--enable-output-avg-walltime,
                 Output ellapsed wall-times),
  [output_avg_walltime="$enableval"],
  [output_avg_walltime=${output_avg_walltime='no'}])
SCI_ARG_ENABLE(ray-scatter,
  AC_HELP_STRING(--ray-scatter,
                 Turn on ray scattering in Arches),
  [ray_scatter="$enableval"],
  [ray_scatter=${ray_scatter=''}])
SCI_ARG_ENABLE(gencode,
  AC_HELP_STRING(--enable-gencode,
                 Specify compute capability for GPU code generation (allowable values: 12, 13, 20, 21, 31)),
  [cuda_gencode="$enableval"],
  [cuda_gencode=${gencode=''}])
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
                 Use SCI memory allocation/deallocation checks.  Note: you cannot use sci-malloc-trace at the same time.),
  [enable_sci_malloc="$enableval"],
  [enable_sci_malloc=${enable_sci_malloc='no'}])
SCI_ARG_ENABLE(memory-init,
  AC_HELP_STRING(--enable-memory-initialization=8_bit_number,
                 Force all new'd and malloc'd memory to be initialied to the value given - defaults to 127.),
  [enable_memory_initialization="$enableval"],
  [enable_memory_initialization=${enable_memory_initialization='no'}])
SCI_ARG_WITH(MallocTrace,
  AC_HELP_STRING(--with-MallocTrace=DIR,
                 Use MallocTrace library.  Note: you cannot use sci-malloc at the same time.),
  [with_malloc_trace="$withval"],
  [with_malloc_trace=""])
SCI_ARG_ENABLE(assertion-level,
  AC_HELP_STRING(--enable-assertion-level=N,
                 Use assertion level N (0-3) (Where 0 turns off assertions, and 3 turns on all assertions.)),
  [enable_assertion_level="$enableval"],
  [enable_assertion_level=""])
SCI_ARG_ENABLE(scinew-line-numbers,
  AC_HELP_STRING(--enable-scinew-line-numbers,
                 Have SCI malloc store the file and line that calls 'scinew'),
  [enable_scinew_line_numbers="yes"],
  [enable_scinew_line_numbers="no"])
SCI_ARG_ENABLE(throw-level,
  AC_HELP_STRING(--enable-throw-level=N,
                 Use throw level N (0-3)),
  [enable_throw_level="$enableval"],
  [enable_throw_level=${enable_throw_level=0}])

SCI_ARG_ENABLE(rath,
  AC_HELP_STRING(--enable-rpath,
                 Use the default runtime library search path),
  [enable_rpath="$enableval"],
  [enable_rpath=${enable_rpath='yes'}])

if test "$enable_sci_malloc" = "yes" -a "$with_malloc_trace" ; then
   AC_MSG_ERROR( "You cannot use both --with-MallocTrace and --enable-sci-malloc!" )
fi

if test "$enable_memory_initialization" != "no"; then
   DEF_MEMORY_INIT="#define INITIALIZE_MEMORY"
   if test "$enable_memory_initialization" = "yes"; then
      memory_init=127
   else
      val=$enable_memory_initialization
      if test $val -lt 128 -a $val -gt 0; then
         memory_init=$val
      else
         AC_MSG_ERROR( "--enable-memory-initialization's value must be from 0 to 127." )
      fi
   fi
   DEF_MEMORY_INIT_NUMBER="#define MEMORY_INIT_NUMBER $memory_init"
fi

AC_SUBST(DEF_MEMORY_INIT)
AC_SUBST(DEF_MEMORY_INIT_NUMBER)
sci_defs_files="$sci_defs_files Core/Malloc/testmem_init.h"

if test "$enable_rpath" = "no"; then
   AC_MSG_WARN("You have removed the rpath flag from some libraries.")
   AC_MSG_WARN("This may have unpredictable affects.")
   AC_MSG_WARN("Do not do this unless you really know what you are doing.")
fi


##  --------------------------------------------------------------------
##  Save off the arguments to configure...
##  --------------------------------------------------------------------

args=`echo "$*" | sed 's% %_%g'`
myargs=`echo $args | sed 's%\^% %g'`
# Setting IFS breaks standard autoconf tests.  
# It thinks we are cross compiling. Don't set it.
#IFS=' '

## Cut off the /configure from the configure command (eg: ../src/configure)
path_to_src=`echo $0 | sed 's/\/configure//'`

if test "$sci_required_petsc" = "yes" -a "$with_petsc" != "built-in"; then
   sci_required_blas=yes
   sci_required_lapack=yes
fi

sci_defs_files="$sci_defs_files include/sci_defs/environment_testdefs.h"

DIRNAME_IS=`type dirname | sed 's%^.* %%'`
BASENAME_IS=`type dirname | sed 's%^.* %%'`

# link this way.
LDRUN_PREFIX="-Wl,-rpath -Wl,"

## We use both of these in a number of spots, so I think this is fair:
## (Eventually we could write our own if necessary...)
if test -z "$DIRNAME_IS" -o -z "$BASENAME_IS"; then
  AC_MSG_ERROR(dirname or basename not found.  Configure script won't work!)
fi


##  --------------------------------------------------------------------
##  -----  determine host type, processor count and OS version  --------
##  --------------------------------------------------------------------

echo 
IS_OSX="no"
IS_WIN="no"
IS_VC="no"
IS_AIX="no"
unset NUM_CPUS

MAKE_ARCHIVES="no"

case $host in
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
    DEF_OSX="#define IS_OSX 1"
    ;;
  *-aix*)
    NUM_CPUS=4
    IS_AIX="yes"
    ;;
  *-mingw*)
    NUM_CPUS=2
    IS_WIN="yes"
    ;;
  *-pc*)
    echo
    AC_MSG_ERROR(Currently Windows only configures from mingw.  See install instructions)
    ;;
  *)
    echo
    AC_MSG_WARN( !!!!!! This system is untested - You are on your own !!!!!!)
    echo
    NUM_CPUS="1"
    ;;
esac

AC_SUBST(IS_OSX)
AC_SUBST(DEF_OSX)
AC_SUBST(IS_WIN)
AC_SUBST(IS_AIX)
sci_defs_files="$sci_defs_files include/sci_defs/osx_testdefs.h"

AC_MSG_CHECKING(physical processor count)
if test -z "$NUM_CPUS"; then
  NUM_CPUS="1"
fi
AC_MSG_RESULT($NUM_CPUS)

##  ----------  determine if this is a 32 or 64 bit machine
AC_LANG([C])

# Apparently AC_PROG_CC will set CFLAGS to "-O2 -g" if you don't specify
# its value first, so... (And AC_CHECK_SIZEOF apparently calls AC_PROG_CC...)
CFLAGS=$CFLAGS

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
  This machine has some strange sizeof(void*): $ac_cv_sizeof_voidp.
  Usually this means that your LD_LIBRARY_PATH is set incorrectly for the 
  compiler you are using.  To verify that this is the problem you will need 
  to copy the C++  program from the end of the config.log file and run the 
  compile command which is just before it in config.log.  Run the executable 
  and see what error message you get. You can also just fix your 
  LD_LIBRARY_PATH if you know what is wrong with it.
  Exiting due to size of void* problem! Read above warnings!])
  ;;
esac

AC_CHECK_SIZEOF(long long)
case $ac_cv_sizeof_long_long in
 8)
  LONG_LONG_SWAP=SWAP_8
  ;;
 4)
  LONG_LONG_SWAP=SWAP_4
  ;;
 *)
  SCI_MSG_ERROR([
  This machine has some strange sizeof(long long): $ac_cv_sizeof_long_long.
  Usually this means that your LD_LIBRARY_PATH is set incorrectly for the 
  compiler you are using.  To verify that this is the problem you will need 
  to copy the C++  program from the end of the config.log file and run the 
  compile command which is just before it in config.log.  Run the executable 
  and see what error message you get. You can also just fix your 
  LD_LIBRARY_PATH if you know what is wrong with it.
  Exiting due to size of long long problem! Read above warnings!])
  ;;
esac

AC_SUBST(LONG_LONG_SWAP)
sci_defs_files="$sci_defs_files Core/Util/testEndian.h"

## --------------

AC_LANG([C++])

if test "$CC" = ""; then
  # Apparently AC_PROG_CC will set CFLAGS to "-O2 -g" if you don't specify
  # its value first, so...
  CFLAGS=$CFLAGS

  AC_PROG_CC(cc gcc)

  case "$host" in 
     *)
        # Check to see if cc is actually gcc
        is_gcc=`$CC --version | grep -i gcc`
        if test "$is_gcc"; then
           if test -f "`which gcc`"; then
              msg="Note, found 'cc' which is actually 'gcc', so"
              AC_MSG_WARN($msg)
              AC_MSG_WARN(we will use 'gcc' for clarity.)
              CC=gcc
           fi
        fi

        ;;
  esac
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
        CXXFLAGS=$CXXFLAGS
        AC_PROG_CXX(CC g++ c++)
        AC_MSG_WARN(C++ compiler not specified with 'CXX' environment variable.\
	         Using default... $CXX)
        ;;
  esac
fi


# Most fortran compilers need an "_" for C function names.
# Some have strange underscore needs.  This flag is used in the
# tools/fspec.pl.in file to generate the correct
# cname of the fortran function.
FORTRAN_UNDERSCORE=end
DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_END"

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

# If the user did not specify the number of bits for build, then default
# to machine default number of bits.
if test "$enable_32bit" = "no" -a "$enable_64bit" = "no"; then
   if test "$NBITS" = 64; then
      enable_64bit="yes"
   else
      enable_32bit="yes"
   fi
fi

AC_MSG_CHECKING(target build word length)
if test "$enable_64bit" = "yes"; then
  AC_MSG_RESULT(64 bits)
  if test "$enable_32bit" = "yes"; then
    SCI_MSG_ERROR(Both 64 and 32 bit compiles specified.)
  fi
else
  enable_32bit="yes"
  AC_MSG_RESULT(32 bits)
fi

##  --------------------------------------------------------------------
##  ----------------------  set variable defaults  ---------------------
##  --------------------------------------------------------------------

##
##  DO NOT "hardcode" ANY libraries (i.e. no -l or -L or -I allowed).
##

if test "$IS_WIN" = "yes"; then
  TIME_IMPL=Time_win32.cc
else
  TIME_IMPL=Time_unix.cc
fi
TEMPLATE_TAG=
TEMPLATE_BOX="<>"
case $host in
     *-darwin*)
	SO_OR_ARCHIVE_EXTENSION=dylib
        if test "$enable_sci_malloc" = "yes"; then
           AC_MSG_ERROR(sci_malloc does not work under mac osx. Though you can try to use: --enable-sci-malloc=force)
           enable_sci_malloc="no"
        elif test "$enable_sci_malloc" = "force"; then
           enable_sci_malloc="yes"
        fi
	;;
	
     *-aix*)
        # On AIX, doesn't automatically generate .so's.  Make .a's and then
        # later we will make .so's from them.
	SO_OR_ARCHIVE_EXTENSION=a
	;;
     *-pc-mingw*)
        SO_OR_ARCHIVE_EXTENSION=dll
        ;;
     *)
	SO_OR_ARCHIVE_EXTENSION=so 
        ;;
esac

THROW_LEVEL="$enable_throw_level"
extra_cflags=""

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
  SCI_MALLOC_ON="yes"
  DEF_MALLOC=""
else
  SCI_MALLOC_ON="no"
  DEF_MALLOC="#define DISABLE_SCI_MALLOC 1"
fi

if test "$with_malloc_trace" ; then
  if test "$IS_OSX" = "yes"; then
      AC_MSG_ERROR(Malloc trace is not supported on Mac OSX...)
  fi
fi

if test "$enable_scinew_line_numbers" = "yes"; then
  DEF_MALLOC_LINENUMBERS="#define USE_TAG_LINENUM 1"
else
  DEF_MALLOC_LINENUMBERS=""
fi

AC_SUBST(NOPRINTDIR)
AC_SUBST(EXENAME)
AC_SUBST(LARGESOS)
AC_SUBST(TIME_IMPL)

AC_SUBST(SCI_MALLOC_ON)
AC_SUBST(DEF_MALLOC)
AC_SUBST(DEF_MALLOC_LINENUMBERS)
sci_defs_files="$sci_defs_files include/sci_defs/malloc_testdefs.h"

AC_SUBST(ASSERTION_LEVEL)
AC_SUBST(THROW_LEVEL)
sci_defs_files="$sci_defs_files include/sci_defs/error_testdefs.h"

#########################################################################

##  ----------------------------------------------------------------------
##  ----------------  set host specific build tools  ---------------------
##  ----------------------------------------------------------------------

DEBUG=yes

AC_MSG_CHECKING(whether debug was enabled)
OGL_ERR_MACRO=""
if test "$debug" = "no"; then
  AC_MSG_RESULT(no)
  DEBUG=no
elif test "$debug" = "yes"; then
  OGL_ERR_MACRO="#define OGL_DBG 1"
  extra_cflags="-g ${extra_cflags}"
  AC_MSG_RESULT(yes (-g) )
else
  OGL_ERR_MACRO="#define OGL_DBG 1"
  extra_cflags="$debug ${extra_cflags}"
  AC_MSG_RESULT(yes ($debug) )
fi

AC_SUBST(DEBUG)
AC_SUBST(OGL_ERR_MACRO)

AC_MSG_CHECKING(whether optimize was enabled)
if test "$optimize" = "no"; then
  if test "$debug" = "no"; then
    # neither debug nor optimize specified, DEFAULT TO: optimized
    extra_cflags="-O2 -DNDEBUG ${extra_cflags}"
    AC_MSG_RESULT(no (default to -O2) )
  else
    # Debug mode specified
    extra_cflags="${no_optimize_flag} ${extra_cflags}"
    # extra_fflags WILL ONLY BE USED WITH g77
    extra_fflags="-ffortran-bounds-check"
    AC_MSG_RESULT(no (${no_optimize_flag}) )
  fi
elif test "$optimize" = "yes"; then
  extra_cflags="-O2 -DNDEBUG ${extra_cflags}"
  AC_MSG_RESULT(yes (-O2) )
else
  extra_cflags="$optimize -DNDEBUG ${extra_cflags}"
  AC_MSG_RESULT(yes ($optimize) )
fi

# Check to see if assertions are off but optimize is on... and warn user if so...
if test "$optimize" != "no"; then

  ASSERTION_LEVEL="$enable_assertion_level"
  if test "$ASSERTION_LEVEL" = ""; then
    add_important_warning "FYI: Configure turned off assertions due to --enable-optimize."
    ASSERTION_LEVEL=0
  else
    if test "$ASSERTION_LEVEL" != "0"; then
      add_important_warning "You explicitly turned on assertions (Level $ASSERTION_LEVEL) but you also specified --enable-optimize.  You can turn assertions off with --enable-assertion-level=0.  Assertions can significantly decrease performance."
    fi
  fi
else
  # Configuring debug build...
  ASSERTION_LEVEL="$enable_assertion_level"
  if test "$ASSERTION_LEVEL" = ""; then
    # Debug builds default to level 3 assertions.
    ASSERTION_LEVEL=3
  fi
fi

if test $ASSERTION_LEVEL != "0" -a $ASSERTION_LEVEL != "1" -a $ASSERTION_LEVEL != "2" -a $ASSERTION_LEVEL != "3"; then
  AC_MSG_ERROR([--enable-assertion-level was set to '$ASSERTION_LEVEL', but must be 0, 1, 2, or 3.])
fi

if test ! -e "`type $CC | sed 's%^.* /%/%'`"; then
  SCI_MSG_ERROR(Could not find the specified C compiler ($CC))
fi
if test ! -e "`type $CXX | sed 's%^.* /%/%'`"; then
  SCI_MSG_ERROR(Could not find the specified C++ compiler (CC))
fi

if test "$IS_WIN" = "yes"; then
  # don't require fortran on windows yet 
  # (Uintah disables fortran compilation on windows for the moment...)
  with_fortran="no"
fi

# enable/disable certain Uintah components
BUILD_ARCHES=yes
BUILD_ICE=yes
BUILD_MPM=yes
BUILD_MODELS_RADIATION=yes
BUILD_WASATCH=no
USE_WASATCH=no
DEF_ARCHES=""
DEF_ICE=""
DEF_MPM=""
DEF_MODELS_RADIATION=""
DEF_WASATCH=""

if test "$USE_ARCHES" = no; then BUILD_ARCHES=no; DEF_ARCHES="#define NO_ARCHES"; fi
if test "$USE_ICE" = no; then BUILD_ICE=no; DEF_ICE="#define NO_ICE"; fi
if test "$USE_MPM" = no; then BUILD_MPM=no; DEF_MPM="#define NO_MPM"; fi
if test "$USE_MODELS_RADIATION" = no; then BUILD_MODELS_RADIATION=no; DEF_MODELS_RADIATION="#define NO_MODELS_RADIATION"; fi
if test "$with_fortran" = no; then 
            BUILD_ARCHES=no;
            DEF_ARCHES="#define NO_ARCHES";
            DEF_NO_FORTRAN="#define NO_FORTRAN";
            NO_FORTRAN="yes";
fi

# NO_FORTRAN is used in configVars.mk[.in] and DEF_NO_FORTRAN is used in uintah_testdefs.h.in

AC_SUBST(BUILD_ARCHES)
AC_SUBST(BUILD_ICE)
AC_SUBST(BUILD_MPM)
AC_SUBST(BUILD_MODELS_RADIATION)
AC_SUBST(DEF_ARCHES)
AC_SUBST(DEF_ICE)
AC_SUBST(DEF_MPM)
AC_SUBST(DEF_NO_FORTRAN)
AC_SUBST(DEF_MODELS_RADIATION)
AC_SUBST(DEF_WASATCH)
AC_SUBST(NO_FORTRAN)

if test "$BUILD_ARCHES" = no -a "$BUILD_MODELS_RADIATION" = no; then
  sci_required_fortran=no
else
  if test "$with_fortran" != no; then
    sci_required_fortran=yes
    if test "$F77" = ""; then
      AC_PROG_F77()
      if test "$F77" = ""; then
        AC_MSG_ERROR(Could not find a fortran compiler!  Please set the F77 environment variable to point to a fortran compiler.)
      fi
      AC_MSG_WARN(Fortran compiler not specified with 'F77' environment variable. Using default... $F77)
    else
      if test ! -e "`type $F77 | sed 's%^.* /%/%'`"; then
        SCI_MSG_ERROR(Could not find the specified Fortran compiler: '$F77' )
      fi
    fi
  fi
fi

##  --  Handle static build stuff... ------------------------

IS_STATIC_BUILD="no"

# If --enable-static, or if lib extension already set to 'a' (eg: AIX):
if test "$static" != "no" -o "$SO_OR_ARCHIVE_EXTENSION" = "a"; then
  AC_MSG_WARN( Building a statically linked version of Uintah! )
  IS_STATIC_BUILD="yes"
  SO_OR_ARCHIVE_EXTENSION=a
  if test "$static" = "full"; then

    # If we don't use the 'full' option, then some shared libs may be
    # linked in.  While we probably should make 'full' the default, this
    # ends up breaking a number of (our test) machines that don't have (all the) static
    # libraries installed.  As 'static' is really used only on specific/large 
    # supercomputers (Crays mostly), and we only/mostly use it on
    # small machines to check for circular dependancies, it is ok to not use
    # '-static' in most cases.

    LDFLAGS="${LDFLAGS} -static"
  fi
  MAKE_ARCHIVES="yes"
  STATIC_BUILD_DEF="#define STATIC_BUILD 1"
else
  SOFLAGS="-shared ${SOFLAGS}"
fi

AC_SUBST(STATIC_BUILD_DEF)
AC_SUBST(MAKE_ARCHIVES)
AC_SUBST(SOFLAGS)
AC_SUBST(IS_STATIC_BUILD)
AC_SUBST(SO_OR_ARCHIVE_EXTENSION)

##  --  search for MallocTrace ------------------------

if test "$with_malloc_trace"; then
  if test "$with_malloc_trace" = "yes"; then
    AC_MSG_ERROR( --with-malloc_trace=DIR requires a Directory be provided. )
  else
    AC_MSG_CHECKING(for MallocTrace files (For now we are just verifying the existence of lib and .h files)) 
    if test ! -d $with_malloc_trace; then
      AC_MSG_ERROR( '$with_malloc_trace' directory does not exist. )
    fi
    if test "$IS_STATIC_BUILD" = "yes"; then
       if test ! -f $with_malloc_trace/lib/libMallocTrace.a; then
       	  AC_MSG_ERROR( '$with_malloc_trace/lib/libMallocTrace.a' does not exist. )
       fi
    else
	if test ! -f $with_malloc_trace/lib/shared/libMallocTrace.so; then
      	   AC_MSG_ERROR( '$with_malloc_trace/lib/shared/libMallocTrace.so' does not exist. )
    	fi
    fi
    if test ! -f $with_malloc_trace/include/MallocTrace.h; then
      AC_MSG_ERROR( '$with_malloc_trace/include/MallocTrace.h' does not exist. )
    fi
    AC_MSG_RESULT(yes)
    DEF_MALLOC_TRACE="#define MALLOC_TRACE 1"
    INC_MALLOC_TRACE_H="-I$with_malloc_trace/include"
    if test "$IS_STATIC_BUILD" = "yes"; then
       MALLOC_TRACE_LIB_DIR_FLAG="$LDRUN_PREFIX$with_malloc_trace/lib -L$with_malloc_trace/lib"
    else								     
       MALLOC_TRACE_LIB_DIR_FLAG="$LDRUN_PREFIX$with_malloc_trace/lib/shared -L$with_malloc_trace/lib/shared"
    fi
    MALLOC_TRACE_LIB_FLAG="-lMallocTrace"
    MALLOC_TRACE_LIB_MPI_FLAG="-lMallocTraceMPI"
  fi
fi
AC_SUBST(DEF_MALLOC_TRACE)
AC_SUBST(INC_MALLOC_TRACE_H)
AC_SUBST(MALLOC_TRACE_LIB_FLAG)
AC_SUBST(MALLOC_TRACE_LIB_DIR_FLAG)
AC_SUBST(MALLOC_TRACE_LIB_MPI_FLAG)

##  --  Check for 64 vs 80 bit floating point number storage in register precision error ------------------------

# In other words, the processor may use 80 bit floating point number registers, and as long as the
# data is kept on the processor (in registers) then you get extra precission... however, as soon as
# the number is stored to memory, it is truncated back to fewer (64?) bits.  At this point, it is possible
# for the original floating pointloatingpoint number to not compare with itself (ie, if the register data
# happens to stay around, and the 'old' value is loaded back from memory and then compared to the register).
# This can cause some very strange and hard to track down errors in the code.  To turn this off, gnu
# provides the -ffloat-store compiler argument.

# WARNING: The following test case was created by Justin Luitjens.  It seems to work, and I can't 
# come up with a better version, but I am not convinced that it will catch the problem in all cases...

cat > conftest-fpr.cc <<EOF

#include <iostream>

int main()
{
  double a=1;
  double b=.01;

  int c=a/b;
  return c != 100; // Shell return codes are backward in that 0 = success, thus the '!='.
}
EOF

$CXX $CXXFLAGS conftest-fpr.cc -o conftest-fpr
./conftest-fpr
if test $? != 0; then
   AC_MSG_WARN( ----------------------------------------------------------------------------------------- )
   AC_MSG_WARN( Floating point - usually 64 vs 80 bit - register precission problem detected. )
   AC_MSG_WARN( Disabling use of floating point registers with '-ffloat-store' compiler flag. )
   AC_MSG_WARN( ----------------------------------------------------------------------------------------- )
   add_important_warning "Automatically added -ffloat-store compiler option to C/CXXFLAGS."
   CFLAGS="$CFLAGS -ffloat-store"
   CXXFLAGS="$CXXFLAGS -ffloat-store"
   # Verify that the compiler accepts the -ffloat-store flag:
   $CXX $CXXFLAGS conftest-fpr.cc -o conftest-fpr 2>&5
   if test $? != 0; then
      rm -f conftest-fpr conftest-fpr.cc
      AC_MSG_ERROR( Tried to add the -ffloat-store flag but the compiler ($CXX) failed to accept it...\nSee 'WARNING' message above. )
   fi
fi

rm -f conftest-fpr conftest-fpr.cc

##  --  search for Zoltan ------------------------

if test "$with_zoltan"; then
  if test "$with_zoltan" = "yes"; then
    AC_MSG_ERROR( --with-zoltan=DIR requires a DIRectory be provided. )
  else
    AC_MSG_CHECKING(for Zoltan files (For now we are just verifying the existence of lib and .h files)) 
    if test ! -d $with_zoltan; then
      AC_MSG_ERROR( '$with_zoltan' directory does not exist. )
    fi
    if test ! -f $with_zoltan/lib/libzoltan.a; then
      AC_MSG_ERROR( '$with_zoltan/lib/libzoltan.a' does not exist. )
    fi
    if test ! -f $with_zoltan/include/zoltan_cpp.h; then
      AC_MSG_ERROR( '$with_zoltan/include/zoltan_cpp.h' does not exist. )
    fi
  fi
  AC_MSG_RESULT(yes)
  DEF_ZOLTAN="#define HAVE_ZOLTAN 1"
  INC_ZOLTAN_H="-I$with_zoltan/include"
  ZOLTAN_LIB_DIR_FLAG="-L$with_zoltan/lib"
  ZOLTAN_LIB_FLAG="-lzoltan"

fi
AC_SUBST(DEF_ZOLTAN)
AC_SUBST(INC_ZOLTAN_H)
AC_SUBST(ZOLTAN_LIB_FLAG)
AC_SUBST(ZOLTAN_LIB_DIR_FLAG)

## -----------------------------------------------

sci_defs_files="$sci_defs_files include/sci_defs/uintah_testdefs.h"

# This flag is for the windows compiler - so it can generate correct
# SHARE flags.
WIN_DLLFLAGS=""
WIN_OTFFLAGS=""

case $host in 
  *-dec*)
    ac_objext=o

    CPP="$CC"
    CXXCPP="$CXX -c"
    ##  this WOFF is required for configure 
    CPPFLAGS=""
    FFLAGS="$extra_cflags $FFLAGS"
    CFLAGS="$extra_cflags -D__USE_STD_IOSTREAM $CFLAGS"
    CXXFLAGS="-model ansi $extra_cflags -D__USE_STD_IOSTREAM $CXXFLAGS"
    if test "$threads" != "none"; then
      CFLAGS="$CFLAGS -pthread"
      CXXFLAGS="$CXXFLAGS -pthread"
    fi
    SOFLAGS="-warning_unresolved -update_registry \$(LIBDIR)/so_locations"
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
    DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_NONE"

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
         # bigtoc helps keep gcc from breaking...
         LDFLAGS="-Wl,-bbigtoc $LDFLAGS"
         ;;
      *)
#      for these compilers: *xlC*) or *KCC*)

         # Tell the compiler that the executable can use a
         # maximum of 1280MB (5*256MB) of memory per processor
         # rather than the default of 256MB.
         LDFLAGS="-bmaxdata:0x50000000 $LDFLAGS"

         if test "$enable_64bit" = "yes"; then
            BIT64FLAG=-q64
            NBITS=64
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

    case $F77 in
      *xlf*)
         FFLAGS="-WF,-D_AIX $extra_cflags $FFLAGS"
         ;;
      *)
         FFLAGS="$extra_cflags $FFLAGS"
         ;;
    esac

    CXXFLAGS="$CFLAGS"

    CPP="$CC -c"
    CXXCPP="$CXX -c"

    ;;
  *-darwin*)
    ac_objext=o
    LDRUN_PREFIX="-L"

#####
# GCC_TEST
#####
    OS_VERSION=`uname -r | cut -f1 -d.`
    if test "$OS_VERSION" -ge 10; then
        AC_MSG_WARN( Running under Snow Leopard or later)
        DEF_OSX_SNOW_LEOPARD_OR_LATER="#define OSX_SNOW_LEOPARD_OR_LATER"
        IS_OSX_SNOW_LEOPARD_OR_LATER="yes"
    fi

    NO_LONG_DOUBLE="-Wno-long-double"
    if test "$IS_OSX_SNOW_LEOPARD_OR_LATER" = "yes"; then
      NO_LONG_DOUBLE=
    fi      

    SCI_CHECK_VERSION($CC,--version,4.0.0,GCC_VERSION=4.0,)

    if test "$GCC_VERSION" = "4.0"; then
       CFLAGS="$CFLAGS $extra_cflags -fno-common -dynamic $NO_LONG_DOUBLE"
    else
       CFLAGS="$CFLAGS $extra_cflags -fno-common -dynamic $NO_LONG_DOUBLE -fcoalesce-templates"
    fi   
    CXXFLAGS="$CXXFLAGS $CFLAGS"

    SOFLAGS="-dynamiclib"

    case $F77 in
      *g77*)
         # extra_fflags has array checking if in debug mode.
         FFLAGS="$FFLAGS $extra_cflags $extra_fflags"

         # put '_' at end and two '_'s if there is a '_' in name.
         # This flag is used by the genfspec.pl script in Uintah (generated
         # from fspec.pl.in).
         FORTRAN_UNDERSCORE=linux
         DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_LINUX"
         ;;
      *gfortran*)
         FFLAGS="$FFLAGS -fbounds-check $extra_cflags"   
         FORTRAN_UNDERSCORE=end
         DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_END"
         ;;        
      *)
         FFLAGS="$FFLAGS $extra_cflags"
         ;;
    esac
    ;;

  *-pc-mingw*)
    case $CXX in
      *cl*)
          ac_objext=obj
          CXXFLAGS="-EHsc -GR"
          
          if test "$debug" = yes; then
            #  no debug information for dynamic compilation yet - the threaded compilations will overwrite the db file
            CFLAGS="-nologo -Od -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_DEPRECATE -Gm -Zi -MD -W1 -Wp64"
            WIN_DLLFLAGS=' -DBUILD_$(subst /,_,$(patsubst %/,%,$(dir $@))) -Fo"$(dir $@)" -Fd"$(dir $@)vc70.pdb"'
            WIN_OTFFLAGS=' -Fd"$(basename $@).pdb"'
          else
            CFLAGS="-nologo -O2 -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_DEPRECATE -O2 -MD -W1"
            WIN_DLLFLAGS=' -DBUILD_$(subst /,_,$(patsubst %/,%,$(dir $@))) -Fo"$(dir $@)"'
          fi
          CXXFLAGS="$CXXFLAGS $CFLAGS"
          WOFF=" -w0"
          SOFLAGS="-link -dll"
          NEED_SONAME=no
          LDRUN_PREFIX="-LIBPATH:"
          IS_VC="yes"
          if !test -f "$PATH_TO_VC"/bin/cl.exe; then
            PATH_TO_VC=""
            echo "invalid PATH_TO_VC"
          fi
          if !test -f "$PATH_TO_PSDK"/Include/windows.h; then
            PATH_TO_PSDK=""
            echo "invalid PATH_TO_PSDK"
          fi
          if !test -f "$PATH_TO_MSYS_BIN"/make.exe; then
            PATH_TO_MSYS_BIN=""
            echo "invalid PATH_TO_MSYS_BIN"
          fi
          
          if test -z "$PATH_TO_VC" -o -z "$PATH_TO_PSDK" -o -z "$PATH_TO_MSYS_BIN"; then
            SCI_MSG_ERROR([One of more of the required VCDIRS variables are not set. Execute
  export PATH_TO_VC=<path-to-vc>
  export PATH_TO_PSDK=<path-to-PlatformSDK>
  export PATH_TO_MSYS_BIN=<msys-bin-dir>
    Use drive letter and forward slashes (/) instead of backslashes (\)
    i.e., C:/msys/1.0/bin])
          fi
          AC_SUBST(PATH_TO_VC)
          AC_SUBST(PATH_TO_PSDK)
          AC_SUBST(PATH_TO_MSYS_BIN)
          ;;
      *g++*)
          ac_objext=o
          CFLAGS=" -Wall"
          CXXFLAGS=" -Wall"
          WIN_DLLFLAGS=' -DBUILD_$(subst /,_,$(patsubst %/,%,$(dir $@)))'
          SCI_CHECK_VERSION($CC,--version,2.95.3,,
            echo configure: error: wrong $CC version.; exit)
          SCI_CHECK_VERSION($CXX,--version,2.95.3,,
            echo configure: error: wrong $CXX version.; exit)
          WOFF=" -w"
          SOFLAGS="-shared ${SOFLAGS}"
          NEED_SONAME=yes
          ;;
      esac
    AC_PROG_CPP
    AC_PROG_CXXCPP
    AC_CHECK_PROG(AS,as,as)
    AC_CHECK_PROG(LD,ld,ld)
    CPPFLAGS="$CPPFLAGS $WOFF"
    CFLAGS="$CFLAGS $extra_cflags"
    CXXFLAGS="$CXXFLAGS $extra_cflags"

    if test "$sci_required_fortran" = "yes"; then
      SCI_CHECK_VERSION($F77,--version,0.5.25,,
        echo configure: error: wrong $F77 version.; exit)
      case $F77 in
        *g77*)
          # extra_fflags has array checking if in debug mode.
          FFLAGS="$FFLAGS $extra_cflags $extra_fflags"

          # put '_' at end and two '_'s if there is a '_' in name.
          # This flag is used by the genfspec.pl script in Uintah (generated
          # from fspec.pl.in).
          FORTRAN_UNDERSCORE=linux
          DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_LINUX"
          ;;
        *gfortran*)
          FFLAGS="$FFLAGS -fbounds-check $extra_cflags"   
          FORTRAN_UNDERSCORE=linux
          DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_LINUX"
          ;;        
        *)
          FFLAGS="$FFLAGS $extra_cflags"
          ;;
      esac
    fi
 
    ;;
  *)
    # linux
    ac_objext=o

    if test "$CC" = "icc"; then
      CFLAGS=" -w1"
      CXXFLAGS=" -w1"
    elif test "$CC" = "pgcc"; then
      # pgcc doesn't support -Wall (at least not on ranger.tacc.utexas.edu under pgcc v7.1-2)
      CFLAGS=
      CXXFLAGS=
    else
      CFLAGS=" -Wall"
      CXXFLAGS=" -Wall"
    fi

    # fPIC option for ppc architecture
    if test `uname -m` = "ppc"; then
       CFLAGS="-fPIC $CFLAGS"
       FFLAGS="-fPIC $FFLAGS"
       CXXFLAGS="-fPIC $CXXFLAGS"
    fi

    if test "$enable_64bit" = "yes"; then
       CFLAGS="-fPIC $CFLAGS"
       FFLAGS="-fPIC $FFLAGS"
       CXXFLAGS="-fPIC $CXXFLAGS"
    fi

    AC_MSG_CHECKING(if special 32/64 bits flags required)
    if test "$enable_32bit" = "yes" -a $NBITS = 64; then
       NBITS=32
       CFLAGS="-m32 $CFLAGS"
       CXXFLAGS="-m32 $CXXFLAGS"
       AC_MSG_RESULT(-m32)
    else 
      if test "$enable_64bit" = "yes" -a $NBITS = 32; then
        NBITS=64
        # Do not set this flag when using intel compilers
        if test "$CC" != "icc"; then
           CFLAGS="-m64 $CFLAGS"
           CXXFLAGS="-m64 $CXXFLAGS"
        fi         
        AC_MSG_RESULT(-m64)
      else 
        AC_MSG_RESULT(no)
      fi
    fi

    if test "$CC" != "pgcc"; then
      SCI_CHECK_VERSION($CC,--version,2.95.3,,
        echo configure: error: wrong $CC version.; exit)

      if test "$sci_required_fortran" = "yes"; then
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

    case `hostname` in
      alc* | mcr*)
        # LLNL's ALC (or MCR) computer is doing something strange with its g77
        # fortran underscore policy to make it compatible with other compilers.
	# We don't need this, so this flag just turns their 'hack' off.
	AC_MSG_WARN(Turning off ALC/MCR's fortran underscore compatibility hack.)
        FFLAGS="$FFLAGS -funderscoring"
        ;;
    esac

    case $F77 in
      *g77*)
         # extra_fflags has array checking if in debug mode.
         FFLAGS="$FFLAGS $extra_cflags $extra_fflags"

         # put '_' at end and two '_'s if there is a '_' in name. 
         # This flag is used by the genfspec.pl script in Uintah (generated
         # from fspec.pl.in).
	 FORTRAN_UNDERSCORE=linux
         DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_LINUX"
         ;;
      *gfortran*)
         FFLAGS="$FFLAGS -fbounds-check $extra_cflags" 
         FORTRAN_UNDERSCORE=end
         DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_END"
         ;;       
      *)
         FFLAGS="$FFLAGS $extra_cflags"
         ;;
    esac

    CXXFLAGS="$CXXFLAGS $extra_cflags"
    NEED_SONAME=yes
    ;;
esac

OBJEXT=$ac_objext

AC_SUBST(DEF_OSX_SNOW_LEOPARD_OR_LATER)
AC_SUBST(NEED_OSX_SYSTEMSTUBS)
AC_SUBST(OBJEXT)

# now that we know which NBITS the user wants, we can save this info
AC_SUBST(NBITS)
sci_defs_files="$sci_defs_files include/sci_defs/bits_testdefs.h"

###

AC_SUBST(WIN_DLLFLAGS)
AC_SUBST(WIN_OTFFLAGS)

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
    AC_MSG_CHECKING( 1 - if depend mode flag '$flag' works for '$CXX')
    rm -f configure-test.o configure-test-dir/configure-test.o
    ac_try="$CXX $CFLAGS -c $flag configure-test-dir/configure-test.d -o configure-test-dir/configure-test.o configure-test-dir/configure-test.cc >/dev/null 2>conftest.out"
    (eval $ac_try) 2>&5
    if test -f configure-test-dir/configure-test.d ; then
      CC_DEPEND_MODE=normal
      CC_DEPEND_REGEN='-MDupdate $(dir $@)depend.mk'
      CC_DEPEND_EXT=depend.mk
    else 
      AC_MSG_RESULT( no )
      echo ""
#      echo "ac_try: $ac_try"   # <= Turn on to debug DEPEND flag issues...
#      echo ""
#      echo "Test results: "
#      echo ""
#      cat conftest.out
#      echo ""
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
      AC_MSG_CHECKING( 2 - if depend mode flag '$flag' works )
      rm -f configure-test.o configure-test-dir/configure-test.o
      ac_try="$CXX $CFLAGS -c $flag configure-test-dir/configure-test.cc -o configure-test-dir/configure-test.o >/dev/null 2>conftest.out"
      (eval $ac_try) 2>&5
      if test -f configure-test-dir/configure-test.$ext ; then
	CC_DEPEND_REGEN=$flag
	CC_DEPEND_EXT=$ext
        if grep "^configure-test-dir/configure-test.o" configure-test-dir/configure-test.$ext > /dev/null; then
          CC_DEPEND_MODE=normal
          break 2
        elif grep "^configure-test.o" configure-test-dir/configure-test.$ext > /dev/null; then
          CC_DEPEND_MODE=modify
          break 2
        fi
      fi
      if test -f configure-test.$ext; then
	CC_DEPEND_REGEN=$flag
	CC_DEPEND_EXT=$ext
        if grep "^configure-test-dir/configure-test.o" configure-test.$ext > /dev/null; then
          CC_DEPEND_MODE=move
        elif grep "^configure-test.o" configure-test.$ext > /dev/null; then
          CC_DEPEND_MODE=modify_and_move
        fi
        break 2
      fi
      AC_MSG_RESULT( no )
    done
  done
fi

# Last resort, try one that spits out to stdout
if test $CC_DEPEND_MODE = unknown; then
  for flag in -M; do
    AC_MSG_CHECKING( 3 - if depend mode flag '$flag' works )
    rm -f configure-test.o configure-test-dir/configure-test.o
    ac_try="$CXX $CFLAGS -c $flag configure-test-dir/configure-test.cc >configure-test-dir/configure-test.d 2>conftest.out"
    (eval $ac_try) 2>&5
    if test -s configure-test-dir/configure-test.d ; then
      AC_MSG_RESULT( stdout )
      CC_DEPEND_MODE=stdout
      CC_DEPEND_REGEN=$flag
      CC_DEPEND_EXT=d
      break
    fi
    AC_MSG_RESULT( no )
  done
fi

if test $IS_WIN = "yes"; then
  # for the moment, don't gen deps on win32
  CC_DEPEND_MODE=normal
  CC_DEPEND_REGEN=""
  CC_DEPEND_EXT=""
fi

if test $CC_DEPEND_MODE = unknown; then
  AC_MSG_ERROR(Cannot determine dependency generation flag. You might have specified bad '--enable-optimize=' flags on your configure line)
fi

AC_MSG_RESULT(yes.  Mode: $CC_DEPEND_MODE.  Ext: $CC_DEPEND_EXT)
rm -f configure-test.*
rm -rf configure-test-dir

if test `basename "$F77"` = "pgf77"; then
  F77_DEPEND_REGEN=-Mcpp=mm
  F77_DEPEND_MODE=stdout
  F77_DEPEND_EXT=d
else
  F77_DEPEND_REGEN=$CC_DEPEND_REGEN
  F77_DEPEND_MODE=$CC_DEPEND_MODE
  F77_DEPEND_EXT=$CC_DEPEND_EXT
fi

AC_SUBST(CC_DEPEND_MODE)
AC_SUBST(CC_DEPEND_REGEN)
AC_SUBST(CC_DEPEND_EXT)
AC_SUBST(F77_DEPEND_MODE)
AC_SUBST(F77_DEPEND_REGEN)
AC_SUBST(F77_DEPEND_EXT)

# Other programs
if test "$IS_WIN" = "yes"; then
  AC_CHECK_PROG(MAKE,make,make,NOT_FOUND,)
else
  AC_PATH_PROG(GMAKE,gmake,NOT_FOUND,,)
  if test "$GMAKE" = "NOT_FOUND"; then
    AC_PATH_PROG(MAKE,make,NOT_FOUND,,)
  else
    MAKE=$GMAKE
  fi
fi
if test "$MAKE" = "NOT_FOUND"; then
  SCI_MSG_ERROR(GNU make not found.)
fi

AC_SUBST(MAKE)
sci_defs_files="$sci_defs_files include/sci_defs/compile_testdefs.h"

SCI_CHECK_VERSION($MAKE,--version,3.78.1,,
  echo configure: error: wrong GNU make version.; exit)

AC_PROG_AWK

## ignore some software tests on windows (for now at least)
if test "$IS_WIN" = "yes"; then
  echo "skipping bison, lex, etags, perl tests"
else

AC_CHECK_PROG(ETAGS, etags, etags)

if test "$sci_required_perl" = "yes" ; then
  AC_CHECK_PROG(PERL, perl, perl, 
    echo configure: error: perl was not found in your PATH.;exit)
else
  AC_CHECK_PROG(PERL, perl, perl)
fi

if test "$PERL" = "perl"; then
  PERL=`type perl | sed 's%^.* %%'`
fi

fi # end if IS_WIN = yes

##  Create the tools dir for later use.
if test "$sci_required_tools" = "yes" ; then
  if test ! -d "tools"; then
    AC_MSG_RESULT(Creating tools directory)
    mkdir -p "tools"
  fi
  if test "$ac_cv_f77_compiler_gnu" = "yes" ; then
     G77=1
  else
     G77=0
  fi
fi

if test "$with_fortran" != no; then 
  AC_MSG_CHECKING(fortran underscore policy)
  AC_MSG_RESULT($FORTRAN_UNDERSCORE)
fi

AC_SUBST(FORTRAN_UNDERSCORE)
AC_SUBST(DEF_FORTRAN_UNDERSCORE)
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
AC_SUBST(ASFLAGS)
AC_SUBST(LDRUN_PREFIX)
AC_SUBST(extra_cflags)
AC_SUBST(MAKE_PARALLELISM)
AC_SUBST(NEED_SONAME)
AC_SUBST(PERL)
AC_SUBST(REPOSITORY_FLAGS)
AC_SUBST(TEMPLATE_TAG)
AC_SUBST(TEMPLATE_BOX)
sci_defs_files="$sci_defs_files include/sci_defs/template_testdefs.h"

##  set the default language after the build tools are found
AC_LANG([C])


#########################################################################
##  Set the no_optimize_flag 
##  

case $host in
  *linux*)
    no_optimize_flag=-O0
    ;;
  *-aix*)

    AC_MSG_WARN(AIX is not completely supported!)
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

AC_LANG([C++])
# Each of the HAVE_??? statements for these header files is found
# in include/sci_defs/config_testedefs.h
AC_CHECK_HEADERS(limits.h sys/select.h unistd.h sys/time.h ext/algorithm stdint.h inttypes.h)

AC_CONFIG_HEADER(include/sci_defs/config_testdefs.h)
sci_defs_files="$sci_defs_files include/sci_defs/config_testdefs.h"

##  ----------------------------------------------------------------------
##  ----------  required checks for required libraries and headers  ------
##  ----------------------------------------------------------------------

##  --  search for the math library  -------------------------------------
AC_LANG(C)

if test "$IS_VC" = "yes"; then
  M_LIB=""
else
  M_LIB=m
fi
SCI_TRY_LINK(M,math libraries and headers,math.h,,$M_LIB,,,[
  double d = fabs(69.05);
],required)

AC_SUBST(M_LIB_FLAG)
AC_SUBST(M_LIB_DIR_FLAG)

##  --  check for the 'rt' library  ---------------------------

SCI_TRY_LINK(RT,rt library,,,rt,,,,optional)
AC_SUBST(RT_LIB_FLAG)

MPI_EXTRA_LIB_FLAG="$MPI_EXTRA_LIB_FLAG $RT_LIB_FLAG"

##  --  check for the required thread library  ---------------------------

SOCKET_LIBRARY=""

case $host in
  *-pc-mingw*)
    threads=win32
    if test $IS_VC = "yes"; then
      SOCKET_LIB_FLAG="ws2_32.lib"
    else
      SOCKET_LIB_FLAG="-lws2_32"
    fi
    ;;
  *)
    if test "$threads" = "unknown"; then
      threads=pthreads
    elif test "$threads" != "pthreads" -a "$threads" != "none"; then
      SCI_MSG_ERROR(unknown thread type for host $host: $threads. exiting.)
    fi
    ;;
esac

case $threads in
   pthreads)
      SCI_TRY_LINK(PTHREAD,pthread library and headers,pthread.h semaphore.h,,
                   pthread,,,,required)

      if test "$HAVE_PTHREAD" = "yes"; then
        THREAD_LIB_FLAG="$PTHREAD_LIB_FLAG"
        DEF_THREAD="#define SCI_PTHREAD 1"
      fi

      AC_MSG_CHECKING(for semaphore library)
      old_LIBS="$LIBS"
      SEMAPHORE_LIB_FLAG=unknown
      for i in "" "$THREAD_LIB_FLAG" $RT_LIB_FLAG; do
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
        if test "$SEMAPHORE_LIB_FLAG" = "$THREAD_LIB_FLAG"; then
          AC_MSG_RESULT(already in thread library: $THREAD_LIB_FLAG)
          SEMAPHORE_LIB_FLAG=""
        elif test "$SEMAPHORE_LIB_FLAG"; then
          AC_MSG_RESULT($SEMAPHORE_LIB_FLAG)
        else
          AC_MSG_RESULT(none required)
        fi
      fi
      LIBS="$old_LIBS"
      ;;
   none)
      DEF_THREAD="#define SCI_NOTHREAD 1"
      AC_MSG_WARN(threads disabled - software may work incorrectly)
      ;;
   win32)
     # for time functions in windows
     DEF_THREAD=
     if test $IS_VC = "yes"; then
       THREAD_LIB_FLAG=winmm.lib
     else
       THREAD_LIB_FLAG=-lwinmm
     fi
     ;;
   *)
      SCI_MSG_ERROR(unknown thread type for host $host: $threads. exiting.)
      ;;
esac      

THREAD_IMPL="Thread_$threads.cc"
AC_SUBST(THREAD_IMPL)

AC_SUBST(THREAD_LIB_FLAG)
AC_SUBST(SOCKET_LIB_FLAG)
AC_SUBST(DEF_THREAD)
sci_defs_files="$sci_defs_files include/sci_defs/thread_testdefs.h"

AC_SUBST(SEMAPHORE_LIB_FLAG)

AC_SUBST(CFLAGS)

##  -- check for gcc atomic support

SCI_TRY_LINK(GCCSYNC,gcc style atomic operations ,,,,,,[ int x;   __sync_add_and_fetch(&x,1);],optional)
if test "$HAVE_GCCSYNC" = "yes"; then
  REFCOUNT_IMPL=RefCounted_gcc.cc
  ATOMIC_IMPL=AtomicCounter_gcc.cc
else
  REFCOUNT_IMPL=RefCounted.cc
  ATOMIC_IMPL=AtomicCounter_default.cc
fi
AC_SUBST(REFCOUNT_IMPL)
AC_SUBST(ATOMIC_IMPL)


##  --  Enable Ray Scatter -----------------------------

if test "$ray_scatter" = "yes"; then
  DEF_RAY_SCATTER="#define RAY_SCATTER 1"
fi
AC_SUBST(DEF_RAY_SCATTER)

##  --  With VisIt -----------------------------

if test "$with_visit"; then

   AC_MSG_CHECKING(for VisIt)

   which cmake > /dev/null 2>&1
   if test $? != 0; then
      AC_MSG_ERROR( VisIt requires CMake. Please make sure 'cmake' is in your path!)
   fi

   if test "$with_visit" = "yes"; then 
      AC_MSG_ERROR( You must provide a DIRECTORY when using --with-visit=DIR. )
   fi
   if ! test -f "$with_visit/bin/xml2cmake"; then 
      AC_MSG_ERROR( Could not find xml2cmake in $with_visit/bin/... )
   fi

   if test -f "./VisIt/udaReaderMTMD/Makefile.visit"; then
      old_visit_dir=`grep TOPDIR ./VisIt/udaReaderMTMD/Makefile.visit`
      if test "$old_visit_dir" != "TOPDIR=$with_visit/src"; then
      	 AC_MSG_WARN(VisIt target directory has changed - removing old VisIt build directory...)
	 rm -rf ./VisIt
      fi
   fi

   CMAKE_LOCATION=`which cmake`

   # Verify that cmake is new enough:
   #
   # Need this ugly hack to handle different types of version strings returned by cmake's different versions... sigh.
   # The first 'sed' removes all alphabetical characters (leaves numbers, spaces, and periods).  The second 'sed'
   # removes leading and trailing space.  The final 'sed' replaces (potential) spaces with a perions (.).
   # Eg: The version string from cmake 2.4.8 is 'cmake version 2.4-patch 8'.
   # Note: autoconf removes one set of [], thus the below sed commands have one extra set of [] around each portion.
   CMAKE_VERSION_STRING=`cmake --version | sed 's/[[^0-9. ]]//g' | sed 's/^[[[:space:]]]*\(.*\)[[[:space:]]]*$/\1/' | sed 's/ /./g'`

   CMAKE_VERSION_MAJOR=`echo "$CMAKE_VERSION_STRING" | awk -F. '{print $1}'`
   CMAKE_VERSION_MINOR=`echo "$CMAKE_VERSION_STRING" | awk -F. '{print $2}'`

   if   test "$CMAKE_VERSION_MAJOR" -ge 3; then
      AC_MSG_WARN(Have not tried using cmake v3.x... It might work...)
   elif test "$CMAKE_VERSION_MAJOR" -lt 2; then
      AC_MSG_ERROR(CMake must be version 2.8 or better.  $CMAKE_LOCATION is version $CMAKE_VERSION_STRING.)
   elif test "$CMAKE_VERSION_MINOR" -lt 8; then
      AC_MSG_ERROR(CMake must be version 2.8 or better.  $CMAKE_LOCATION is version $CMAKE_VERSION_STRING.)
   fi

   sci_defs_files="$sci_defs_files VisIt/udaReaderMTMD/avtudaReaderMTMDFileFormat.C"

   BUILD_VISIT=yes
   UINTAH_UDA_TO_VIS_LIB=`pwd`/lib/libStandAlone_tools_uda2vis.$SO_OR_ARCHIVE_EXTENSION
   VISIT_INSTALL_DIR=$with_visit
   AC_MSG_RESULT(found)
fi

AC_SUBST(BUILD_VISIT)
AC_SUBST(UINTAH_UDA_TO_VIS_LIB)
AC_SUBST(VISIT_INSTALL_DIR)

##  --  search for OFED (Infiniband Drivers) -----------------------------

if test "$with_ofed"; then

   if test "$with_ofed" = "yes"; then
      AC_MSG_ERROR( You must specify a directory with the '--with-ofed=DIR' option! )
   else
      dir="$with_ofed"
      BASE_LIB_PATH(dir,$dir)

      SCI_TRY_LINK( OFED,Infiniband stuff (for MPI),,,
                    ibverbs ibumad rdmacm pthread,$dir,,,required)
      MPI_EXTRA_LIB_FLAG="$MPI_EXTRA_LIB_FLAG $OFED_LIB_FLAG"
      MPI_EXTRA_LIB_DIR="$MPI_EXTRA_LIB_DIR $OFED_LIB_DIR"
   fi
fi

##  --  search for MX (myrinet drivers) -----------------------------

if test "$with_mx"; then

   if test "$with_mx" = "yes"; then
      AC_MSG_ERROR( You must specify a directory with the '--with-mx=DIR' option! )
   else
      dir="$with_mx"
      BASE_LIB_PATH(dir,$dir)

      SCI_TRY_LINK( MX,Myrinet stuff (for MPI),,,myriexpress,$dir,,,required)
      MPI_EXTRA_LIB_FLAG="$MPI_EXTRA_LIB_FLAG $MX_LIB_FLAG"
      MPI_EXTRA_LIB_DIR="$MPI_EXTRA_LIB_DIR $MX_LIB_DIR"
   fi
fi


##  --  search for dlopen  -----------------------------------------------
SCI_TRY_LINK(DL,dlopen in libc,dlfcn.h,,c,,,[
  dlopen("bogus", 0);
],optional)

if test "$HAVE_DL" = "yes"; then
   HAVE_LIB_DL="yes"
else
  case $host in
  *-darwin*)
    SCI_TRY_LINK(DL,dlopen in libdl,dlfcn.h,$sci_thirdparty_include,dl,$sci_thirdparty_lib,,[
      dlopen("bogus", 0);
    ],required)
    ;;
  *-pc-mingw*)
    echo "Using Windows dynamic loader"
    DL_LIB_FLAG=""
    DL_LIB_DIR_FLAG=""
    ;;
  *)
    SCI_TRY_LINK(DL,dlopen in libdl,dlfcn.h,,dl,,,[
      dlopen("bogus", 0);
    ],required)
    ;;
  esac
fi

AC_SUBST(DL_LIB_FLAG)
AC_SUBST(DL_LIB_DIR_FLAG)

##  --  search for required headers  --------------------------------------

##  --  optional stat64 headers
# stat64 is a CC lib so it has to be checked as a CC file not C.
ext="$ac_ext"
ac_ext="cc"

SCI_TRY_LINK(STAT64,sys/stat.h,sys/stat.h,,,,,[
  struct stat64 buf;
],optional)

# Restore the previous settings.
ac_ext="$ext"

if test "$HAVE_STAT64" = "yes"; then
  DEF_STAT64="#define HAVE_STAT64 1"
fi

AC_SUBST(DEF_STAT64)
sci_defs_files="$sci_defs_files include/sci_defs/stat64_testdefs.h"  

##  --  required standard headers
SCI_TRY_LINK(STD_HDRS,standard headers,stdio.h stdlib.h stddef.h stdarg.h string.h limits.h errno.h fcntl.h float.h memory.h ctype.h signal.h locale.h assert.h sys/types.h sys/stat.h sys/timeb.h,,,,,,required)

# don't test for malloc.h or values.h on darwin
# don't test a bunch of then on win32
case $host in
  *-darwin*)
    SCI_TRY_LINK(STD_HDRS, standard headers-2,unistd.h dirent.h sys/time.h sys/param.h netdb.h sys/select.h sys/socket.h sys/mman.h sys/ioctl.h sys/resource.h sys/wait.h sys/utsname.h rpc/types.h netinet/in.h,,,,,,required)
    ;;
  *-pc-mingw*)
    ;;

  *)
    # default system headers that cannot be tested on all systems
    SCI_TRY_LINK(STD_HDRS,standard headers-2,unistd.h dirent.h sys/time.h sys/param.h values.h malloc.h netdb.h sys/select.h sys/socket.h sys/mman.h sys/ioctl.h sys/resource.h sys/wait.h sys/utsname.h rpc/types.h netinet/in.h,,,,,,required)
    ;;
esac

AC_SUBST(DEF_IEEEFP)
sci_defs_files="$sci_defs_files include/sci_defs/ieeefp_testdefs.h"

##  --  required STL headers 
##  --      (only look for STL after going into cplusplus mode)
##
AC_LANG(C++)

SCI_TRY_LINK(STL_HEADERS,standard template library headers,algorithm vector map iostream string,,,,,[
  std::vector<int> vec;
  std::cout << "blah" << std::endl;
],required)

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
# win32 has hash_map in stdext namespace (7 deprecates it in std, 8 requires stdext)
if test "$DEF_HASHMAP" = "None"; then
  AC_TRY_COMPILE([#include <hash_map>
                 stdext::hash_map<int, int> xx;], ,
		 [DEF_HASHMAP=HAVE_STDEXT_HASHMAP])
fi
# gcc-4.3 has depricated <ext/hash_map> so check for <backward/hash_map> first
if test "$DEF_HASHMAP" = "None" && test "$IS_OSX" != "yes"; then
  AC_TRY_COMPILE([#include <tr1/unordered_map>
                 std::tr1::unordered_map<int, int> xx;], ,
		 [DEF_HASHMAP=HAVE_TR1_HASHMAP])
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
  DEF_HASHMAP="#define $DEF_HASHMAP 1"
fi
AC_SUBST(DEF_HASHMAP)
sci_defs_files="$sci_defs_files include/sci_defs/hashmap_testdefs.h"
AC_LANG(C)

##  --  search for libxml2 --
_sci_optional=required
_sci_lib=


if test "$with_libxml2"; then
   # If --with-libxml2 has a parameter, then use it
   _sci_lib="$with_libxml2"
   BASE_LIB_PATH(_sci_lib,$_sci_lib)
   _sci_optional=specific
fi

if test "$with_libxml2" = "built-in"; then
  XML2_LIB_FLAG="-lxml2"
  AC_MSG_RESULT( "Using built in libxml2... no explicit linking will be done." )
else

  AC_MSG_CHECKING(for $_sci_optional libxml2)
  if test ! -e $_sci_lib/../bin/xml2-config; then
     AC_MSG_RESULT(Not found under $_sci_lib)
     AC_MSG_CHECKING(for xml2-config in your path)
     XML2_DIR=`xml2-config --prefix`
     if test -e "$XML2_DIR/lib"; then
        _sci_lib=$XML2_DIR/lib
        AC_MSG_RESULT(xml2-config found...)
     else
        AC_MSG_RESULT(xml2-config program not found...)
     fi
  fi

  found_xml2_config="no"

  if test -e $_sci_lib/../bin/xml2-config; then
      found_xml2_config="yes"
      AC_MSG_WARN(Using $_sci_lib/../bin/xml2-config to set libxml2 variables.)

      xml2_static=`xml2-config | grep static`

      if test "$IS_STATIC_BUILD" = "yes" -a "$IS_OSX" != "yes"; then
        if test -n "$xml2_static"; then
           STATIC_FLAG=--static
        else
           STATIC_FLAG=''
        fi
      fi


      XML2_LIB_DIR_FLAG=`$_sci_lib/../bin/xml2-config --libs $STATIC_FLAG`
      XML2_LDFLAGS=""
      XML2_LIB_PATH=""
      for i in $XML2_LIB_DIR_FLAG;do
        is_lib_path=`echo $i | sed -n -e "s/-L//p"`
        if test -z "$is_lib_path";then
          XML2_LDFLAGS="$XML2_LDFLAGS $i"
        else 
          if ! test $i = "-L/usr/lib";then
            XML2_LIB_PATH="$XML2_LIB_PATH $i"
          fi
        fi
      done
      includes=`$_sci_lib/../bin/xml2-config --cflags`
  fi
  SCI_TRY_LINK(XML2,xml2 library and headers,libxml/tree.h libxml/parser.h,
                $includes,$XML2_LDFLAGS,$XML2_LIB_PATH,,[
  #if defined(LIBXML_TREE_ENABLED) && defined(LIBXML_OUTPUT_ENABLED)
  xmlDocPtr doc = NULL;       /* document pointer */
  xmlNodePtr root_node = NULL, node = NULL, node1 = NULL;/* node pointers */
  xmlDtdPtr dtd = NULL;       /* DTD pointer */
  char buff[256];
  int i, j;

  LIBXML_TEST_VERSION;

  /*
    * Creates a new document, a node and set it as a root node
    */
  doc = xmlNewDoc(BAD_CAST "1.0");
  root_node = xmlNewNode(NULL, BAD_CAST "root");
  xmlDocSetRootElement(doc, root_node);
  #else
  //create compile error so configure fails
  WrongVersionOrNotInstalled *x = 0;
  #endif
  ],required)

  DEF_XML2="#define HAVE_XML2 1"
  break

  if test ! "$DEF_XML2"; then
    echo
    AC_MSG_ERROR(One or more of the xml2 components is missing.)
  fi
fi

AC_SUBST(XML2_LIB_FLAG)
AC_SUBST(XML2_LIB_DIR_FLAG)
AC_SUBST(INC_XML2_H)

##  ----------------------------------------------------------------------
##  ----------  required checks for optional libraries and headers  ------
##  ----------------------------------------------------------------------

##  --  search for optional headers  ------------------------------------
AC_LANG(C++)
SCI_TRY_LINK(SSTREAM,sstream header,sstream,,,,,,optional)

if test "$HAVE_SSTREAM" = "yes"; then
   SSTREAM_COMPAT="no"
else
   # Use SCIRun sstream compatibility files...
   SSTREAM_COMPAT="yes"
fi

AC_SUBST(SSTREAM_COMPAT)
AC_LANG(C)
##  --  search for SGI'S or Win32's traceback library  ------------------------------ 

_sci_optional=optional
if test "$with_exc"; then
  _sci_lib="$with_exc"
  _sci_inc="$with_exc/../include"
  _sci_optional=specific
else
  _sci_lib=
  _sci_inc=
fi

if test "$IS_WIN" = "yes"; then
  # it's 'imagehlp' but call it EXC anyway so we don't have to change
  # other config parameters
  SCI_TRY_LINK(EXC,WIN32 traceback library and headers,windows.h imagehlp.h,$_sci_inc,imagehlp psapi,$_sci_lib,,,required)
else
  SCI_TRY_LINK(EXC,traceback library and headers,libexc.h,$_sci_inc,exc,$_sci_lib,,,$_sci_optional)
fi

if test "$HAVE_EXC" = "yes" -a "$IS_WIN" != "yes"; then
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
sci_defs_files="$sci_defs_files include/sci_defs/exe_testdefs.h"

##  --  search for gzopen (and related)  ----------------------------------

## Since variables create by this section are used to check for PNG
## and TEEM stuff this section needs to preceed them.

_sci_optional=required
_sci_lib=
_sci_inc=

if test "$with_zlib"; then
  if test "$with_zlib" = "yes"; then
    AC_MSG_WARN(Zlib is already required - you do not need to specify --with-zlib.)
  else
    _sci_lib="$with_zlib/lib"
    _sci_inc="$with_zlib/include"
    _sci_optional=specific
  fi
fi

SCI_TRY_LINK(Z,zlib library and header,zlib.h,$_sci_inc,z,$_sci_lib,,,$_sci_optional)

if test "$HAVE_Z" = "yes"; then
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
sci_defs_files="$sci_defs_files include/sci_defs/z_testdefs.h"

##  --  search for jpeg and tiff -----------------------------------------

_sci_optional=optional
_sci_lib=
_sci_inc=

if test "$with_jpeg"; then
   if test "$with_jpeg" = "yes"; then
      _sci_optional=required
   else
      _sci_lib="$with_jpeg/lib"
      _sci_inc="$with_jpeg/include"
      _sci_optional=specific
   fi
fi

SCI_TRY_LINK(JPEG,jpeg library,,$_sci_inc,jpeg,$_sci_lib,,,$_sci_optional)

AC_SUBST(JPEG_LIB_FLAG)
AC_SUBST(JPEG_LIB_DIR_FLAG)

_sci_optional=optional
_sci_lib=
_sci_inc=

if test "$with_tiff"; then
   if test "$with_tiff" = "yes"; then
      _sci_optional=required
   else
      _sci_lib="$with_tiff/lib"
      _sci_inc="$with_tiff/include"
      _sci_optional=specific
   fi
fi

SCI_TRY_LINK(TIFF,tiff library,tiff.h,$_sci_inc,tiff,$_sci_lib,$M_LIB_DIR_FLAG $M_LIB_FLAG,,$_sci_optional)

AC_SUBST(TIFF_LIB_FLAG)
AC_SUBST(TIFF_LIB_DIR_FLAG)
AC_SUBST(HAVE_TIFF)

##  --  search for png --------------------------------------

# Default values:
_sci_optional=optional
_sci_lib=
_sci_inc=

if test "$with_png" = "yes"; then
  # If --with-png is specified without a parameter, then it makes it required
  _sci_optional=required
else 
  if test "$with_png"; then
    # If --with-png has a parameter, then use it
    #   The BASE_LIB_PATH below will add /lib or /lib{64,32} to the sci_lib
    _sci_lib="$with_png" 
    _sci_inc="$with_png/include"
    BASE_LIB_PATH(_sci_lib,$_sci_lib)
    _sci_optional=specific
  fi 
fi

SCI_TRY_LINK(PNG,png library and headers,png.h,$_sci_inc,png,$_sci_lib,$Z_LIB_DIR_FLAG $Z_LIB_FLAG,,optional)

# Now check for the correct version.
if test "$HAVE_PNG" = "yes"; then

    AC_MSG_CHECKING(for PNG version 1.0.2 or better.)
    cat > conftest-png.c <<EOF

         #include <png.h>
         #include <stdio.h>

         int main() {

           if (PNG_LIBPNG_VER < 10002) {
	       printf("1\n");
	   } else {
	       printf("0\n");
           }	   
          return 0;
         }
EOF
    #echo "$CC $INC_PNG_H conftest-png.c -o conftest-png"
    $CC $INC_PNG_H conftest-png.c -o conftest-png
    #ls -l conftest-png
    png_error=`./conftest-png`
    rm -f conftest-png conftest-png.c

    if test "$png_error" != "0"; then
       AC_MSG_RESULT(PNG library too old.  Need at least 1.0.2.  Turning off PNG.)
       DEF_PNG=""
       HAVE_PNG="no"
       PNG_LIB_FLAG=""
       PNG_LIB_DIR_FLAG=""
       INC_PNG_H=""
       DEF_PNG=""
    else
       AC_MSG_RESULT(yes.)
       DEF_PNG="#define HAVE_PNG 1"
    fi

    break
fi
 
if test "$HAVE_PNG" = "no"; then
  # Try looking in a sci specific location (but only if it exists)
  if test -d /usr/sci/local; then
    BASE_LIB_PATH(png_lib_dir,/usr/sci/local)
    SCI_TRY_LINK(PNG,png library and headers (SCI),png.h,/usr/sci/local/include,png,
                 $png_lib_dir,$Z_LIB_DIR_FLAG $Z_LIB_FLAG,,optional)
  fi
else
  if test "$_sci_optional" = "specific"; then
       AC_MSG_CHECKING(that PNG is in the requested location '$with_png/lib')
       # verify that it was found in the location requested.
       if test ! -e $with_png/lib/libpng.so &&
          test ! -e $with_png/lib/libpng.a &&
          test ! -e $with_png/lib/libpng.dylib; then
            echo
            AC_MSG_ERROR(Found libpng but not in requested directory: $with_png/lib)
       fi
       AC_MSG_RESULT(yes.)
    fi
fi

if test "$with_png" && test ! "$DEF_PNG"; then
  echo
  AC_MSG_ERROR(One or more of the png components is missing.)
  echo
fi

AC_SUBST(PNG_LIB_FLAG)
AC_SUBST(PNG_LIB_DIR_FLAG)
AC_SUBST(INC_PNG_H)
AC_SUBST(DEF_PNG)
AC_SUBST(HAVE_PNG)

##  --  check if exceptions work properly --------------------------------

AC_MSG_CHECKING(for working exceptions.)
cat > conftest-exc.cc <<EOF

    #include <stdio.h>

    int main() {       
      try {
        throw "hello world";        
      }
      catch( const char* e ) {
        printf("error is %s\n", e);
      }
    }
EOF
    $CXX -o conftest-exc conftest-exc.cc
    ./conftest-exc > /dev/null 2>&1 

    if test "$?" != "0"; then
       AC_MSG_RESULT(not found)
       enable_exceptions_crash=1
       EXCEPTIONS_CRASH_DEF="#define EXCEPTIONS_CRASH 1"
    else
       AC_MSG_RESULT(yes)
       enable_exceptions_crash=0
       EXCEPTIONS_CRASH_DEF=""
    fi

    rm -f conftest-exc.cc conftest-exc
AC_SUBST(EXCEPTIONS_CRASH_DEF)

##  --  search for mpeg  -------------------------------------------------

_sci_optional=optional
if test "$with_mpeg"; then
  _sci_lib="$with_mpeg/lib"
  _sci_optional=specific
fi

SCI_TRY_LINK(MPEGE,MPEG library,,,mpege,$_sci_lib,,,$_sci_optional)

if test "$HAVE_MPEGE" = "yes"; then
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

sci_defs_files="$sci_defs_files include/sci_defs/image_testdefs.h"

##  --  search for scisock --------------------------------------

AC_SUBST(SCISOCK_LIB_FLAG)
AC_SUBST(SCISOCK_LIB_DIR_FLAG)
AC_SUBST(INC_SCISOCK_H)


sci_defs_files="$sci_defs_files include/sci_defs/scisock_testdefs.h"

##  --  search for unp --------------------------------------

if test "$with_unp" = "yes"; then
  AC_MSG_ERROR(Please provide the location of the UNP package: --with-unp=<dir>)
fi

if test "$with_unp"; then
  _sci_lib="$with_unp/lib"
  _sci_inc="$with_unp/include"
fi

SCI_TRY_LINK(UNP,unp socket library and headers,extern_C unpv12e/unp.h unpv12e/unpthread.h,$_sci_inc,
             unp scisock,$_sci_lib,,[
  Connect(0, NULL, 0);
],optional)

AC_SUBST(UNP_LIB_FLAG)
AC_SUBST(UNP_LIB_DIR_FLAG)
AC_SUBST(INC_UNP_H)

##  --  search for the fortran library  ----------------------------------
# Always check for fortran, since we now always try to enable blas

F_LIB_FLAG=""
F_LIB_DIR_FLAG=""

_sci_lib=
_sci_inc=
_sci_optional=optional

if test "$sci_required_fortran" = "yes"; then
  _sci_optional=required
fi

if test "$with_fortran" = "yes"; then
  _sci_optional=required
elif test "$with_fortran" -a "$with_fortran" != "no"; then
  _sci_lib="$with_fortran/lib"
  _sci_inc="$with_fortran/include"
  _sci_optional=specific
fi

if test "$with_fortran" != "no"; then
  case $host in 
    *aix*)
      # AIX Fortran Library Check
      SCI_TRY_LINK(F,fortran libraries,,,xlf90,$_sci_lib,,,$_sci_optional)
      ;;
    *linux*)
      # Linux Library Check
      case $F77 in
        *pgf77*)
          SCI_TRY_LINK(F,fortran libraries,,,pgftnrtl,$_sci_lib,$RT_LIB_FLAG,,$_sci_optional)
          ;;
        *gfortran*)
          SCI_TRY_LINK(F,fortran libraries,,,gfortran,$_sci_lib,,,$_sci_optional)
          ;;
        *ifort*)
          SCI_TRY_LINK(F,fortran libraries,,,ifcore,$_sci_lib,,,$_sci_optional)
          ;;
        *ftn*)
          SCI_TRY_LINK(F,fortran libraries,,,,$_sci_lib,,,$_sci_optional)
          ;;
        *)
          SCI_TRY_LINK(F,fortran libraries,,,g2c,$_sci_lib,,,$_sci_optional)
          ;;
       esac
      ;;
    *dec*)
      SCI_TRY_LINK(F,fortran libraries,,,for Ufor,$_sci_lib,,,$_sci_optional)
      ;;
    *-darwin*)
      # OSX Library Check
      if test "$F77" = "xlf"; then
        SCI_TRY_LINK(F,Mac fortran library - xlf90,,$_sci_inc,xlf90,$_sci_lib,,,$_sci_optional)
      else
        SCI_TRY_LINK(F,Mac fortran library - gfortran,,$_sci_inc,gfortran,$_sci_lib,,,optional)
        if test "$HAVE_F" = "no"; then
             SCI_TRY_LINK(F,mac fortran library - g2c,,$_sci_inc,g2c,$_sci_lib,,,$_sci_optional)
        fi
      fi
      ;;
    *-pc-mingw*)
      # Mingw g77 and libg2c
      SCI_TRY_LINK(F,fortran libraries,,,g2c,$_sci_lib,,,$_sci_optional)
      ;;
    *)
      SCI_TRY_LINK(F,fortran libraries,,,F77,$_sci_lib,,,$_sci_optional)
      ;;
  esac
fi
  
AC_SUBST(F_LIB_FLAG)
AC_SUBST(F_LIB_DIR_FLAG)

MPI_EXTRA_LIB_FLAG="$MPI_EXTRA_LIB_FLAG $F_LIB_FLAG"
MPI_EXTRA_LIB_DIR="$MPI_EXTRA_LIB_DIR $F_LIB_DIR_FLAG"

AC_LANG(C)

# Clean up all the fortran flags that don't have individual without-fortran guards on them...
if test "$with_fortran" = no; then 
  FFLAGS="fortran_disabled";
  extra_fflags="fortran_disabled";
  FORTRAN_UNDERSCORE="// FORTRAN_UNDERSCORE => fortran_disabled";
  DEF_FORTRAN_UNDERSCORE="// DEF_FORTRAN_UNDERSCORE => fortran_disabled";
fi

##  --  Electric Fence  -------------------------------------

if test "$with_electric_fence"; then

  if test "$with_electric_fence" = "yes"; then
    SCI_TRY_LINK(ELECTRIC_FENCE,
                 electric fence library,,,
                 efence,
                 ,,,
                 required)
  else
    SCI_TRY_LINK(ELECTRIC_FENCE,
                 electric fence library,,,
                 efence,$with_electric_fence,
                 ,,
                 specific)
  fi
fi

AC_SUBST(HAVE_ELECTRIC_FENCE)
AC_SUBST(ELECTRIC_FENCE_LIB_FLAG)
AC_SUBST(ELECTRIC_FENCE_LIB_DIR_FLAG)

##  --  search for acml (optimized blas/lapack) stuff  -------------------------------------

if test "$with_acml"; then

  if test "$with_blas"; then
    AC_MSG_ERROR(You cannot use both --with-blas and --with-acml.)
  fi
  if test "$with_atlas"; then
    AC_MSG_ERROR(You cannot use both --with-atlas and --with-acml.)
  fi
  if test "$with_mkl"; then
    AC_MSG_ERROR(You cannot use both --with-mkl and --with-acml.)
  fi

  if test "$with_acml" = "yes"; then
    _sci_lib=
    _sci_inc=
    _sci_optional=required
  elif test "$with_acml"; then
    _sci_lib="$with_acml/lib"
    _sci_inc="$with_acml/include"
    _sci_optional=specific
  fi

  case $host in
    *-darwin*)
       AC_MSG_ERROR(We recommend that you don't use ACML on a Mac... use the default vecLib instead.)
       ;;
     *)
       SCI_TRY_LINK(BLAS,acml version of (blas/lapack) libraries and headers,acml.h,
                    $_sci_inc,acml acml_mv,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,,$_sci_optional)
       ;;
  esac

  HAVE_LAPACK=$HAVE_BLAS

  DEF_LAPACK="#define HAVE_LAPACK 1"
  DEF_BLAS="#define HAVE_BLAS 1"

  sci_required_blas=no
fi

##  --  search for mkl (optimized blas/lapack) stuff  -------------------------------------

if test "$with_mkl"; then

  if test "$with_blas"; then
    AC_MSG_ERROR(You cannot use both --with-blas and --with-mkl.)
  fi
  if test "$with_atlas"; then
    AC_MSG_ERROR(You cannot use both --with-atlas and --with-mkl.)
  fi

  if test "$with_mkl" = "yes"; then
    AC_MSG_ERROR(You must specify a directory with --with-mkl=DIR)
  elif test "$with_mkl"; then
    _sci_lib="$with_mkl/lib"
    _sci_inc="$with_mkl/include"
    _sci_optional=specific
  fi

  case $host in
    *-darwin*)
       AC_MSG_ERROR(We recommend that you don't use MKL on a Mac... use the default vecLib instead.)
       ;;
     *)
       # Newer version of MKL doesn't have libguide, so need two separate checks...

       if test -e $_sci_lib/libguide.$SO_OR_ARCHIVE_EXTENSION; then
         SCI_TRY_LINK(BLAS,1 - mkl version of (blas/lapack) libraries and headers,mkl_cblas.h mkl_lapack.h,
                      $_sci_inc,mkl mkl_core mkl_intel_lp64 guide,$_sci_lib,
                      $THREAD_LIB_FLAG $M_LIB_DIR_FLAG $M_LIB_FLAG,[
                                                                    float x[]={0.7f};
                                                                    float y[]={0.8f};
                                                                    float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                                    ],$_sci_optional)
       else
         SCI_TRY_LINK(BLAS,2 - mkl version of (blas/lapack) libraries and headers,mkl_cblas.h mkl_lapack.h,
                      $_sci_inc,mkl mkl_core mkl_intel_lp64,$_sci_lib,
                      $THREAD_LIB_FLAG $M_LIB_DIR_FLAG $M_LIB_FLAG,[
                                                                    float x[]={0.7f};
                                                                    float y[]={0.8f};
                                                                    float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                                    ],$_sci_optional)
       fi

       ;;
  esac

  HAVE_LAPACK=$HAVE_BLAS

  CBLAS_H="mkl_cblas.h"
  HAVE_CBLAS=$HAVE_BLAS
  DEF_CBLAS="#define HAVE_CBLAS 1"
  DEF_BLAS="#define HAVE_BLAS 1"
  DEF_LAPACK="#define HAVE_LAPACK 1"
  sci_required_blas=no
fi

##  --  search for atlas (optimized blas/lapack) stuff  -------------------------------------

if test ! "$HAVE_BLAS"; then
  # MKL test above did not find mkl... so look for atlas

  if test "$with_atlas" -a "$with_blas"; then
    AC_MSG_ERROR(You cannot use both --with-blas and --with-atlas.)
  fi

  _sci_lib=
  _sci_inc=
  _sci_optional=optional

  if test "$with_atlas" = "yes"; then
    _sci_optional=required
  elif test "$with_atlas"; then
    _sci_lib="$with_atlas/lib"
    _sci_inc="$with_atlas/include"
    _sci_optional=specific
  fi

  case $host in
    *-darwin*)
       if test "$with_atlas" && test "$with_atlas" != yes ; then
         AC_MSG_ERROR(Don't specify a path for atlas when on a Mac (we use vecLib by default.))
       fi
       SCI_TRY_LINK(BLAS,atlas version of libraries and headers (OSX),vecLib/cblas.h,
                    -faltivec,,,-framework vecLib,[
                                                   float x[]={0.7f};
                                                   float y[]={0.8f};
                                                   float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                   ],$_sci_optional)
       ;;
     *)
       SCI_TRY_LINK(BLAS,atlas version of libraries and headers,cblas.h,
                    $_sci_inc,cblas f77blas atlas,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,[
                                                                                         float x[]={0.7f};
                                                                                         float y[]={0.8f};
                                                                                         float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                                                         ],$_sci_optional)
       ;;
  esac

  HAVE_CBLAS=$HAVE_BLAS
  DEF_CBLAS=$DEF_BLAS

  if test "$HAVE_BLAS" = "yes"; then
    DEF_BLAS="#define HAVE_BLAS 1"
    DEF_CBLAS="#define HAVE_CBLAS 1"
    CBLAS_H="cblas.h"
    sci_required_blas=no
  fi
fi

AC_SUBST(CBLAS_H)


##  --  search for blas stuff  -------------------------------------------

_sci_lib=
_sci_inc=

if test "$HAVE_BLAS" != "yes"; then
  # if HAVE_BLAS is yes at this point, then atlas has been found and we
  # don't want to check for blas.

  if test "$with_blas" && test "$with_blas" != "yes"; then
    _sci_lib="$with_blas/lib"
    _sci_inc="$with_blas/include"
    _sci_specific=yes
  fi
  
  case $host in
    *aix*)
       SCI_TRY_LINK(BLAS,essl library,,,
                      essl,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,,required)
       ;;
    *dec*)
       SCI_TRY_LINK(BLAS,cxml library,,,
                      cxml,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,,required)
       ;;
    *)
       # Can't use the "specific" key word as we are testing for a number
       # of different configurations.
       SCI_TRY_LINK(BLAS,blas and cblas libraries-1,cblas.h,
                    $_sci_inc,cblas blas,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,[
                                                                                float x[]={0.7f};
                                                                                float y[]={0.8f};
                                                                                float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                                                ],optional)
       if test "$HAVE_BLAS" = "yes"; then
          HAVE_CBLAS=yes
          CBLAS_H="cblas.h"
          DEF_CBLAS="#define HAVE_CBLAS 1"
       else
          SCI_TRY_LINK(BLAS,blas and scs libraries-2,cblas.h,
            $_sci_inc,blas scs_mp,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,[
                                                                         float x[]={0.7f};
                                                                         float y[]={0.8f};
                                                                         float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                                         ],optional)
          if test "$HAVE_BLAS" = "no"; then
             #scs library should be used on the SGI altix
             #SCI_TRY_LINK(BLAS,blas and scs libraries-2,cblas.h,
             #   $_sci_inc,blas scs_mp,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,,optional)
             SCI_TRY_LINK(BLAS,blas and cblas libraries-2,cblas.h,
               $_sci_inc,blas,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,[
                                                                     float x[]={0.7f};
                                                                     float y[]={0.8f};
                                                                     float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                                     ],optional)
             if test "$HAVE_BLAS" = "yes"; then
                HAVE_CBLAS=yes
                CBLAS_H="cblas.h"
                DEF_CBLAS="#define HAVE_CBLAS 1"
             else
                SCI_TRY_LINK(BLAS,blas library,,,blas,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,,optional)
                if test "$HAVE_BLAS" = "no"; then
	           SCI_TRY_LINK(BLAS,optional fblas library,,,
		   	     fblas,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,,optional)
                else
                   # Verify it was in the right spot
                   if test "$_sci_specific" = "yes"; then
                     SCI_TRY_LINK(BLAS,blas library,,,blas,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,,specific)
                   fi
                fi
             fi
          else
             HAVE_CBLAS=yes
             CBLAS_H="cblas.h"
             DEF_CBLAS="#define HAVE_CBLAS 1"
          fi
       fi
       ;;
  esac

  if test "$HAVE_BLAS" = "yes"; then
    DEF_BLAS="#define HAVE_BLAS 1"
  fi

  if test "$with_blas" && test ! "$DEF_BLAS"; then
    echo
    AC_MSG_ERROR(One or more of the BLAS components is missing.)
    echo
  fi

fi # end sci_required_blas

# these are set above by atlas check if atlas check succeeded
AC_SUBST(HAVE_BLAS)
AC_SUBST(HAVE_CBLAS)
AC_SUBST(INC_BLAS_H)
AC_SUBST(BLAS_LIB_FLAG)
AC_SUBST(BLAS_LIB_DIR_FLAG)
AC_SUBST(DEF_BLAS)
AC_SUBST(DEF_CBLAS)
sci_defs_files="$sci_defs_files include/sci_defs/blas_testdefs.h"

##  --  search for lapack stuff  -----------------------------------------

if test "$HAVE_LAPACK" != "yes"; then
  # if HAVE_LAPACK is yes at this point, then atlas/mkl has been found and we
  # don't want to check for it here.

  if test "$with_lapack"; then
     _sci_lib="$with_lapack/lib"
  else
     _sci_lib="$LIB_DIR_BLAS"
  fi

  case $host in
    *-darwin*)
       SCI_TRY_LINK(LAPACK,lapack version oflibraries and headers,
                    vecLib/clapack.h,,,,-faltivec -framework vecLib,,optional)
       if test "$HAVE_LAPACK" = yes ; then
          LAPACK_LIB_FLAG="-faltivec -framework vecLib"
       fi
       ;;
     *)
       SCI_TRY_LINK(LAPACK,lapack library,,,lapack,$_sci_lib,$BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG,,optional)
       ;;
  esac

  if test "$HAVE_LAPACK" != "yes"; then
    SCI_TRY_LINK(LAPACK,flapack library,,,flapack,$_sci_lib,$BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG,,optional)
  fi

  # In some cases, some files need complib.sgimath_mp for the lapack
  # library.  I have created this new var LAPACKMP to keep track of this
  # information.  If the configure script finds complib.sgimath_mp, then
  # the LAPACKMP vars will be updated.  If it does not, then the LAPACK
  # vars will be used.
  HAVE_LAPACKMP=$HAVE_LAPACK
  LAPACKMP_LIB_FLAG=$LAPACK_LIB_FLAG
  LAPACKMP_LIB_DIR_FLAG=$LAPACK_LIB_DIR_FLAG

  if test "$HAVE_LAPACK" != "yes"; then
    SCI_TRY_LINK(LAPACK,complib.sgimath library,,,complib.sgimath,$_sci_lib,,,optional)
    if test "$HAVE_LAPACK" = "yes"; then
         AC_MSG_WARN(Using complib for lapack)
         SCI_TRY_LINK(LAPACKMP,complib.sgimath_mp library,,,complib.sgimath_mp,$_sci_lib,,,optional)
    fi
  fi

  if test "$HAVE_LAPACK" = "yes"; then
    DEF_LAPACK="#define HAVE_LAPACK 1"
  fi

  if test "$sci_required_lapack" = "yes" -o "$with_lapack" && test ! "$DEF_LAPACK"; then
    echo
    AC_MSG_WARN(Could not find LAPACK in $_sci_lib)
    AC_MSG_ERROR(The required LAPACK component is missing.)
    echo
  fi

fi

AC_SUBST(HAVE_LAPACK)
AC_SUBST(DEF_LAPACK)
AC_SUBST(LAPACK_LIB_FLAG)
AC_SUBST(LAPACK_LIB_DIR_FLAG)

AC_SUBST(HAVE_LAPACKMP)
AC_SUBST(LAPACKMP_LIB_FLAG)
AC_SUBST(LAPACKMP_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/lapack_testdefs.h"

##  --  allow for extra mpi libraries  -----------------------------------

# If --with-mpi-extra-libs is set, then it will be one or more
# absolute paths to additionaly libraries necessary to link MPI.  This
# code parses the given library (libraries) and adds (through the
# MPI_EXTRA flags) to the MPI test link line.

if test "$with_mpi_extra_libs"; then
   the_libs=$(echo $with_mpi_extra_libs | tr "," "\n")
   for lib in $the_libs; do
      if test ! -f "$lib"; then
         AC_MSG_ERROR(Extra MPI lib '$lib' does not exist...)
      else
         the_dir=`dirname $lib`
         if test "$the_dir" = "."; then
            AC_MSG_ERROR(Extra MPI lib '$lib' must include an absolute path...)
         fi
         the_base=`basename $lib | cut -f"1" -d"." | sed 's/lib//g'` 
         MPI_EXTRA_LIB_FLAG="$MPI_EXTRA_LIB_FLAG -l$the_base"
         if test "$the_dir" != "/usr/lib" -a "$the_dir" != "/usr/lib64"; then
            # Don't add /usr/lib or /usr/lib64 to the path...
            MPI_EXTRA_LIB_DIR="$MPI_EXTRA_LIB_DIR -L$the_dir"
         fi
      fi
   done
fi

##  --  search for mpi  --------------------------------------------------
sci_required_mpi="yes"
if test "$sci_required_mpi" = "yes"; then
  echo
  AS_MESSAGE(checking for required MPI components......)
  echo

  if test "$with_mpi" -a \( "$with_mpi_include" -o "$with_mpi_lib" \); then
    AC_MSG_ERROR( [Please use either --with-mpi or --with-mpi-lib/include, but not both.] )
  fi
  if test "$with_mpi_include" = "yes"; then
    AC_MSG_ERROR( [--with-mpi-include=DIR must specify a DIRectory.] )
  fi
  if test "$with_mpi_lib" = "yes"; then
    AC_MSG_ERROR( [--with-mpi-lib=DIR must specify a DIRectory.] )
  fi
  if test \( "$with_mpi_lib" -a "$with_mpi_include" = "" \) -o \( "$with_mpi_lib" = "" -a "$with_mpi_include" \); then
    AC_MSG_ERROR( [Both --with-mpi-lib=DIR and --with-mpi-include=DIR must be specified...] )
  fi
  if test "$with_mpi_include"; then
    if ! test -d "$with_mpi_include"; then
      AC_MSG_ERROR( [The directory specified with --with-mpi-include=DIR is not valid: $with_mpi_include] )
    fi
    if ! test -d "$with_mpi_lib"; then
      AC_MSG_ERROR( [The directory specified with --with-mpi-lib=DIR is not valid: $with_mpi_lib] )
    fi
  fi

  case $host in
  #  *aix*)
        # On LLNL blue pacific, the "mp" compilers (build_scripts) automagically
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
        _sci_optional=optional
        if test "$with_mpi"; then
           if test "$with_mpi" = "yes"; then
              _sci_optional=required
           else
              _sci_optional=specific
              _sci_inc="$with_mpi/include"
              # Account for lib64 or lib32 or just lib...
              BASE_LIB_PATH(_sci_lib,$with_mpi)
           fi
        else
          if test "$with_mpi_lib"; then
            _sci_inc="$with_mpi_include"
            _sci_lib="$with_mpi_lib"
          fi
        fi

	# After each attempt to locate mpi, we need to verify that the mpi it
	# chose is in the path we specified.  This is because some machines
	# have MPI installed in a standard location and sometimes will choose
	# that one over the one we specified.
	the_file=libmpi
	
        # /usr/include/mpi is the standard location of mpi on some linux dists.
        SCI_TRY_LINK(MPI,mpi library and header-1,mpi.h,
                     $_sci_inc /usr/include/mpi,mpi lam util pmpi $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,,,optional)

        if test "$HAVE_MPI" = "yes" && test "$_sci_optional" = "specific" &&
           test ! -e $_sci_lib/$the_file.so &&
           test ! -e $_sci_lib/$the_file.a &&
           test ! -e $_sci_lib/$the_file.dylib; then
             AC_MSG_WARN(Found libmpi but not in requested directory: $_sci_lib)
	     HAVE_MPI="no"
        fi

        if test "$HAVE_MPI" != "yes"; then
          # Try without pmpi (and with $THREAD_LIB_FLAG) for the newer version
          # of lam mpi.
          # Use -DMPIPP_H so we don't include mpi++.h where we don't want it.

          SCI_TRY_LINK(MPI,mpi library and header-2,mpi.h,
                       $_sci_inc /usr/include/mpi,mpi lam util $THREAD_LIB_FLAG $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,-DMPIPP_H,,optional)

          # verify that it was found in the location requested.
          if test "$HAVE_MPI" = "yes" && test "$_sci_optional" = "specific" &&
             test ! -e $_sci_lib/$the_file.so &&
             test ! -e $_sci_lib/$the_file.a &&
             test ! -e $_sci_lib/$the_file.dylib; then
             AC_MSG_WARN(Found libmpi but not in requested directory: $_sci_lib)
	     HAVE_MPI="no"
          fi

          if test "$HAVE_MPI" != "yes"; then
             # If building with newer LAM on OSX, pthreads and util not needed.
             SCI_TRY_LINK(MPI,mpi library and header-3,mpi.h,
                          $_sci_inc /usr/include/mpi,lam mpi $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,-DMPIPP_H,,optional)

             # verify that it was found in the location requested
             if test "$HAVE_MPI" = "yes" && test "$_sci_optional" = "specific" &&
                test ! -e $_sci_lib/$the_file.so &&
                test ! -e $_sci_lib/$the_file.a &&
                test ! -e $_sci_lib/$the_file.dylib; then
                AC_MSG_WARN(Found libmpi but not in the requested directory: $_sci_lib)
                HAVE_MPI="no"
             fi                        

             if test "$HAVE_MPI" != "yes"; then
                SCI_TRY_LINK(MPI,mpi library and header-4,mpi.h,
                             $_sci_inc /usr/include/mpi,mpi $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,-DMPIPP_H,,optional)

                # verify that it was found in the location requested.
                if test "$HAVE_MPI" = "yes" && test "$_sci_optional" = "specific" &&
                   test ! -e $_sci_lib/$the_file.so &&
                   test ! -e $_sci_lib/$the_file.a &&
                   test ! -e $_sci_lib/$the_file.dylib; then
                   AC_MSG_WARN(Found libmpi but not in requested directory: $_sci_lib)
                   HAVE_MPI="no"
                fi

                if test "$HAVE_MPI" != "yes"; then
		   # try mpich with mpich_threadm and mpl library
                   SCI_TRY_LINK(MPI,1. mpich library and header-2,mpi.h,
                                  $_sci_inc /usr/include/mpi,mpich_threadm mpl $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,,,optional)

                   # try mpich_threadm again without mpl library 
                   if test "$HAVE_MPI" != "yes"; then
                      SCI_TRY_LINK(MPI,2. mpich library and header,mpi.h,
                                  $_sci_inc /usr/include/mpi,mpich_threadm $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,,,optional)
                   fi
                   # try mpich with mpl library  
                   if test "$HAVE_MPI" != "yes"; then
                      SCI_TRY_LINK(MPI,3. mpich library and header,mpi.h,
                                  $_sci_inc /usr/include/mpi,mpich mpl $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,,,optional)

                   fi        
                   # try just plain mpich  
                   if test "$HAVE_MPI" != "yes"; then
                      SCI_TRY_LINK(MPI,4. mpich library and header,mpi.h,
                                  $_sci_inc /usr/include/mpi,mpich $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,,,optional)
                   fi
                   # try mpich again with pmpich
                   if test "$HAVE_MPI" != "yes"; then
                      SCI_TRY_LINK(MPI,5. mpich library and header,mpi.h,
                                  $_sci_inc /usr/include/mpi,mpich pmpich $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,,,optional)
                   fi
                   if test "$HAVE_MPI" = "yes"; then
                      HAVE_MPICH="yes"
                   fi
                fi
             fi
          fi
        fi
        
        if test "$HAVE_MPI" = "yes"; then
	   if test "$HAVE_MPICH" != "yes"; then
             ########################################################################
             ## Determine if mpi is actually mpich...
             cat > conftest-mpich.cc <<EOF
#include <mpi.h>
int main(int argc, char **argv)
{
  char * s = MPICH_VERSION;
  return 0;
}
EOF
             $CC $INC_MPI_H -c conftest-mpich.cc > /dev/null 2>&1
             if test -f conftest-mpich.o; then
		 AC_MSG_WARN(Note: found what looked like mpi but was actually mpich.)
                 HAVE_MPICH=yes
             fi
             rm -rf conftest-mpich*
             ########################################################################
           fi

           if test "$IS_WIN" = "yes"; then
             header=""
           else
             header="lib"
           fi
           if test "$_sci_optional" = "specific"; then
              the_file="$header"mpi
              if test "$HAVE_MPICH" = "yes"; then
                the_file="$header"mpich
              fi

              # verify that it was found in the location requested.
              if test ! -e $_sci_lib/$the_file.so &&
                 test ! -e $_sci_lib/$the_file.a &&
                 test ! -e $_sci_lib/$the_file.lib &&
                 test ! -e $_sci_lib/$the_file.dylib; then
                   echo
                   AC_MSG_ERROR(Found libmpi but not in requested directory: $_sci_lib)
              fi
           fi

           if test "$HAVE_MPICH" = "yes"; then
             DEF_MPI="#define HAVE_MPICH 1"
           else
             DEF_MPI="#define HAVE_MPI 1"
           fi
        fi

        if test "$with_mpi" && test ! "$DEF_MPI"; then
          echo
          AC_MSG_ERROR(One or more of the MPI components is missing.)
          echo
        fi
	
  esac
  
  if test ! "$DEF_MPI"; then
     echo
     SCI_MSG_ERROR(one or more of the MPI components is missing.)
  fi

  MPI_LIB_FLAG="$MPI_LIB_FLAG $MALLOC_TRACE_LIB_MPI_FLAG"
  MPI_LIB_DIR_FLAG="$MPI_LIB_DIR_FLAG $MALLOC_TRACE_LIB_DIR_FLAG"
fi #end sci_required_mpi

AC_SUBST(HAVE_MPI)
AC_SUBST(DEF_MPI)
AC_SUBST(INC_MPI_H)
AC_SUBST(MPI_LIB_FLAG)
AC_SUBST(MPI_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/mpi_testdefs.h"

#  --  search for PETSc  --------------------------------------------
if test "$with_petsc" = "no"; then # Handle --without option
  AC_MSG_WARN( "PETSc disabled." )
else  

  if test "$with_petsc" = "built-in"; then
     HAVE_PETSC="yes"
     DEF_PETSC="#define HAVE_PETSC 1"
  else
    # Make sure we have either mpi, blas, and lapack
    # (It seems that under AIX and OSF, the libs that cover blas also cover lapack)
    if (test "$HAVE_MPI" = "yes" ) && test "$HAVE_BLAS" = "yes" && 
         (test "$PETSC_ARCH" = "aix5.1.0.0" || test "$PETSC_ARCH" = "dec" || test "$HAVE_LAPACK" = "yes"); then
      _sci_lib_dir=

      if test "$with_petsc"; then
         if test "$with_petsc" = "yes"; then
            AC_MSG_ERROR(--with-petsc=DIR requires a DIRectory to be specified.)
         fi
         version_file=$with_petsc/include/petscversion.h
         _sci_inc="$with_petsc/include"
      else
         version_file=/usr/include/petsc/petscversion.h
         lib_dir=/usr/lib
         _sci_inc=/usr/include/petsc
         BASE_LIB_PATH(_sci_lib_dir,$lib_dir)
      fi

      if test -f "$version_file"; then

        petsc_version_major=`grep PETSC_VERSION_MAJOR $version_file | grep define | awk '{print $3}'`
        petsc_version_minor=`grep PETSC_VERSION_MINOR $version_file | grep define | awk '{print $3}'`

        if test "$petsc_version_major" = "2"; then
          if test -z "$PETSC_ARCH"; then
            AC_MSG_ERROR(For PETSC version 2x - please set PETSC_ARCH.)
          fi
	  AC_MSG_WARN(Looking for PETSc arch: $PETSC_ARCH)
          if test -z "$_sci_lib_dir"; then
            _sci_lib_dir="$with_petsc/lib/$PETSC_ARCH"
            _sci_inc="$_sci_inc $with_petsc/bmake/$PETSC_ARCH"
          fi
        else
          if test "$petsc_version_major" = "3"; then
            if test -z "$_sci_lib_dir"; then
               _sci_lib_dir="$with_petsc/lib"
            fi
            if test "$PETSC_ARCH"; then
               AC_MSG_ERROR(For PETSC version 3x - do NOT set PETSC_ARCH.)
            fi
          else
            AC_MSG_ERROR(Could not determine PETSc version number...)
          fi
        fi

        SCI_TRY_LINK( PETSC,
                      petsc libraries and headers,
                      petsc.h,
                      $_sci_inc,
                      petsc petscvec petscmat petscdm petscsnes petscts petscksp,
                      $_sci_lib_dir,
                      $INC_MPI_H -DMPIPP_H $LAPACK_LIB_DIR_FLAG $LAPACK_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $MALLOC_TRACE_LIB_DIR_FLAG $MALLOC_TRACE_LIB_FLAG $M_LIB_FLAG $MPI_LIB_DIR_FLAG $MPI_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG $DL_LIB_FLAG $THREAD_LIB_FLAG,
                      ,
                      optional)

        # 
        if test "$HAVE_PETSC" != "yes"; then
                SCI_TRY_LINK(PETSC,
                     petsc library and header,
                     petsc.h,
                     $_sci_inc,
                     petsc,
                     $_sci_lib_dir,
                     $INC_MPI_H -DMPIPP_H $LAPACK_LIB_DIR_FLAG $LAPACK_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $MALLOC_TRACE_LIB_DIR_FLAG $MALLOC_TRACE_LIB_FLAG $M_LIB_FLAG $MPI_LIB_DIR_FLAG $MPI_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG $DL_LIB_FLAG $THREAD_LIB_FLAG,
                     ,
                     required)
        fi

        if test "$HAVE_PETSC" != "yes" ; then
          AC_MSG_ERROR(Could not find PETSc...)
        fi
        DEF_PETSC="#define HAVE_PETSC 1"
      else
        AC_MSG_WARN(Could not find PETSc version file... Skipping.)
      fi
    else
      AC_MSG_WARN(One or more of MPI or BLAS or LAPACK is missing.)
      AC_MSG_WARN(HAVE_MPI: $HAVE_MPI)
      AC_MSG_WARN(HAVE_BLAS: $HAVE_BLAS)
      AC_MSG_WARN(HAVE_LAPACK: $HAVE_LAPACK)
      AC_MSG_WARN(Not configuring PETSc.)
    fi
  fi
fi

AC_SUBST(HAVE_PETSC)
AC_SUBST(PETSC_LIB_FLAG)
AC_SUBST(PETSC_LIB_DIR_FLAG)
AC_SUBST(DEF_PETSC)
AC_SUBST(INC_PETSC_H)

sci_defs_files="$sci_defs_files include/sci_defs/petsc_testdefs.h"


##  --  search for xalan  ---------------------------------------------------
if test "$sci_required_xalan" ; then
  AC_MSG_CHECKING(for xalan-j.)

  if test "$with_xalan"; then
	dir="$with_xalan"
  fi
  if test -a $dir/bin/xalan.jar -a $dir/bin/xml-apis.jar -a $dir/bin/xercesImpl.jar; then
	AC_MSG_RESULT($dir)
	XALAN_PATH="$dir/bin/xalan.jar:$dir/bin/xml-apis.jar:$dir/bin/xercesImpl.jar"
        DEF_XALAN="#define XALAN 1"
  else	
	AC_MSG_RESULT(not found)
  fi

  if test "$with_xalana" && test ! "$DEF_XALAN"; then
    echo
    AC_MSG_ERROR(One or more of the XALAN components is missing.)
    echo
  fi

fi

AC_SUBST(XALAN_PATH)

##  --  if building uintah, search for TAU makefile  ---------------------

if test "$sci_required_tau" = "yes"; then
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
include $with_tau
all:
	\$(TAU_CXX) -o conftest-tau-a.out \$(TAU_DEFS) \$(TAU_INCLUDE) conftest-tau.cc \$(TAU_SHLIBS) \$(TAU_MPI_LIBS)
EOF

  ac_try="$MAKE TAU_MAKEFILE=$with_tau -f conftest-tau-makefile > conftestout1 2> conftesterr1"
  (eval $ac_try) 2>&5
  ac_status=$?
  rm -f conftest-tau.cc conftest-tau-makefile conftest-tau-a.out conftestout1 conftesterr1
  if test "$ac_status" != 0; then
    SCI_MSG_ERROR(TAU libraries not working. Bye!)
  fi

  TAU_MAKEFILE="$with_tau"
fi

AC_SUBST(TAU_MAKEFILE)

##  --  search for papi stuff  -------------------------------------------
_sci_lib=
_sci_inc=
_sci_optional=optional
if test "$sci_required_papi" = "yes"; then
 
  if test "$with_papi" = "yes"; then
    _sci_optional=required
  else
    _sci_optional=specific
    _sci_lib="$with_papi"
    BASE_LIB_PATH(_sci_lib,$_sci_lib)
    _sci_inc="$with_papi/include"
  fi
 
  AC_LANG(C++)
  SCI_TRY_LINK(PAPI, PAPI libraries and headers,
               papi.h,
               $_sci_inc,
               papi,
               $_sci_lib,
               ,
               [long long flops;
               int event=PAPI_DP_OPS;
               float x=0.0;
               PAPI_start_counters(&event, 1);
               x++;
               PAPI_stop_counters(&flops, 1);],
               $_sci_optional)
  AC_LANG(C)     
  
  DEF_PAPI="#define USE_PAPI_COUNTERS 1"

fi #end sci_required_papi

# these are all defined after the call to SCI_COMPILE_LINK_TEST
AC_SUBST(HAVE_PAPI)
AC_SUBST(DEF_PAPI)
AC_SUBST(INC_PAPI_H)
AC_SUBST(PAPI_LIB_DIR_FLAG)
AC_SUBST(PAPI_LIB_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/papi_testdefs.h"


##  --  search for google perftools stuff  -------------------------------------------
_sci_lib=
_sci_inc=
_sci_optional=optional
if test "$sci_required_gperftools" = "yes"; then

  if test "$with_gperftools" = "yes"; then
    _sci_optional=required
  else
    _sci_optional=specific
    _sci_lib="$with_gperftools"
    # account for lib64 or lib32 or just lib...
    BASE_LIB_PATH(_sci_lib,$_sci_lib)
    _sci_inc="$with_gperftools/include"
  fi
  
  
  if test "$IS_OSX" = "yes"; then
     gperf_lib_list=tcmalloc_and_profiler
  else
     gperf_lib_list="tcmalloc profiler unwind"
  fi

  AC_LANG(C++)
  SCI_TRY_LINK(GPERFTOOLS, Google Perftools libraries and headers,
              google/heap-profiler.h google/profiler.h,
              $_sci_inc,
              $gperf_lib_list,
              $_sci_lib,
              ,
              [ProfilerStart("prefixc");
              HeapProfilerStart("prefixm");
              ProfilerStop();
              HeapProfilerStop();],
              $_sci_optional)
  AC_LANG(C)

  DEF_GPERFTOOLS="#define USE_GPERFTOOLS 1"

fi #end sci_required_gperftools

AC_SUBST(HAVE_GPERFTOOLS)
AC_SUBST(DEF_GPERFTOOLS)
AC_SUBST(INC_GPERFTOOLS_H)
AC_SUBST(GPERFTOOLS_LIB_DIR_FLAG)
AC_SUBST(GPERFTOOLS_LIB_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/gperftools_testdefs.h"


##  --  search for freetype stuff  -------------------------------------------
# Default values:
_sci_lib=
_sci_inc=
_sci_optional=optional

if test "$with_freetype" = "yes"; then
  # If --with-freetype is specified without a parameter, then it makes it required
  _sci_optional=required
elif test "$with_freetype"; then
  # If --with-freetype has a parameter, then use it
  _sci_lib="$with_freetype/lib"
  _sci_inc="$with_freetype/include"
  _sci_optional=specific
fi

AC_MSG_CHECKING(for $_sci_optional freetype)

found_freetype_config="no"

if test -z "$_sci_lib"; then
   freetype_bin=/usr/bin
else
   freetype_bin=$_sci_lib/../bin
fi

if test -e $freetype_bin/freetype-config; then
    found_freetype_config="yes"
    echo
    AC_MSG_RESULT(Using $freetype_bin/freetype-config to set freetype variables.)

    includes=`$freetype_bin/freetype-config --cflags`
elif test -e $sci_thirdparty_dir/bin/freetype-config; then
    AC_MSG_RESULT(Using $sci_thirdparty_dir/bin/freetype-config to set freetype variables.)
    found_freetype_config="yes"
    includes=`$sci_thirdparty_dir/bin/freetype-config --cflags`
    _sci_lib="$sci_thirdparty_lib"
else
    AC_MSG_RESULT(No 'freetype-config' file found in $freetype_bin.)
    # try the 3P for windows
    _sci_lib="$sci_thirdparty_lib"
    _sci_inc="$sci_thirdparty_include"
    includes=""
fi
SCI_TRY_LINK(FREETYPE,freetype library and headers,ft2build.h,
             $_sci_inc,freetype,$_sci_lib,$includes,,$_sci_optional)

if test "$HAVE_FREETYPE" = "yes"; then
  INC_FREETYPE_H="$INC_FREETYPE_H $includes"
  DEF_FREETYPE="#define HAVE_FREETYPE 1"
elif test "$found_freetype_config" = "no"; then
   if test "$_sci_optional" != "optional"; then
     echo
     AC_MSG_WARN(Cannot find a freetype-config file to get freetype variables.)
     AC_MSG_WARN(This file usually resides in freetypelib_dir/../bin/freetype-config.)
     AC_MSG_WARN(Please install freetype and then reconfigure.)
   fi
fi

if test "$with_freetype" && test ! "$DEF_FREETYPE"; then
  echo
  AC_MSG_ERROR(One or more of the freetype components is missing.)
fi

AC_SUBST(FREETYPE_LIB_FLAG)
AC_SUBST(FREETYPE_LIB_DIR_FLAG)
AC_SUBST(DEF_FREETYPE)
AC_SUBST(INC_FREETYPE_H)

##  --  search for teem  -------------------------------------------------

# Default values: (Really required, but need to give info to user if fails.)
#                 (See below.)
_sci_optional=optional
_sci_lib=
_sci_inc=

if test "$with_teem"; then
  if test "$with_teem" = "yes"; then
    AC_MSG_WARN(Teem is already required - you do not need to specify --with-teem.)
  else
    # If --with-teem has a parameter, then use it
    _sci_lib="$with_teem"
    _sci_inc="$with_teem/include"
    BASE_LIB_PATH(_sci_lib,$_sci_lib)
    _sci_optional=required
  fi 

  # Determine if teem is using png.  Assume teem is either in the 3rd Party
  # or where the user has specified (as we are going to use nm to look at its
  # symbols and are not trying to link against it (at this point.))

  result=`nm $_sci_lib/libteem.so | grep png_write_info`
  if test _sci_optional != "optional" && test -n "$result" && test "$HAVE_PNG" != "yes"; then
    AC_MSG_WARN(Teem was built with PNG support.  However the Uintah configure)
    AC_MSG_WARN(did not find PNG.  You need to determine where it is... as)
    AC_MSG_WARN(Teem seems to have found it already.  You will most likely then need)
    AC_MSG_WARN(to use --with-png=/dir/above/dir/having/libpng on your configure line.)
    AC_MSG_ERROR(Please fix this problem and then configure again.)
  fi

  SCI_TRY_LINK(TEEM,teem library and headers,teem/nrrd.h teem/ten.h,$_sci_inc,teem,$_sci_lib,$M_LIB_FLAG,[
               float a[3], b[9], c[7];
               tenEigensolve_f(a,b,c);
               ]
               ,$_sci_optional)

  if test "$HAVE_TEEM" = "yes"; then
     DEF_TEEM="#define HAVE_TEEM 1"
  fi
fi

AC_SUBST(HAVE_TEEM)
AC_SUBST(DEF_TEEM)
AC_SUBST(INC_TEEM_H)
AC_SUBST(TEEM_LIB_FLAG)
AC_SUBST(TEEM_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/teem_testdefs.h"

##  --  search for hdf5  -------------------------------------------------
AC_LANG(C++)
# The sci_required_hdf5 could have been set to yes when using the
# with_hdf5 agument or from a package in the aclocal.m4 it.
# These are the only two locations that the sci_required_* should be set.

if test "$sci_required_hdf5" = "yes"; then 

  _sci_optional=required
  if test "$with_hdf5"; then
    if test "$with_hdf5" != "yes"; then
      _sci_lib="$with_hdf5/lib"			# user defined directory
      _sci_inc="$with_hdf5/include"
      _sci_optional=specific
    fi
  fi

  SCI_TRY_LINK(HDF5,hdf5 library and headers,hdf5.h H5Cpp.h,$_sci_inc,hdf5 hdf5_cpp,$_sci_lib,,,$_sci_optional)

  # Set the DEF_HDF5 var which is placed in the hdf5_testdefs.h.in
  # Incude hdf5_defs.h in your code to optionally use the library.
  if test "$HAVE_HDF5" = "yes"; then
     DEF_HDF5="#define HAVE_HDF5 1"
  fi
  
fi #end sci_required_hdf5

AC_SUBST(DEF_HDF5)		# Place this var in the hdf5_testdefs.h.in
AC_SUBST(INC_HDF5_H)		# Place this var in the configVars.mk.in
AC_SUBST(HDF5_LIB_FLAG)		# Place this var in the configVars.mk.in
AC_SUBST(HDF5_LIB_DIR_FLAG)	# Place this var in the configVars.mk.in
sci_defs_files="$sci_defs_files include/sci_defs/hdf5_testdefs.h"
AC_LANG(C)

##  --  search for Boost ------------------------
AC_LANG([C++])
if test "$with_boost"; then
   if test "$with_boost" = "yes"; then
       AC_MSG_ERROR( --with-boost=DIR requires a DIRectory be provided. )
   elif test ! -d $with_boost; then
        echo ""
        AC_MSG_ERROR( '$with_boost' directory does not exist. )
   elif test "$with_boost" = "/usr"; then
        # Check for header files
        BOOST_INCLUDE_DIR=$with_boost/include/boost
        BOOST_LIBRARY_DIR=$with_boost/lib
 
   else
        BOOST_INCLUDE_DIR=$with_boost/include
        BOOST_LIBRARY_DIR=$with_boost/lib
   
   fi
   
    _boost_incl="boost/signal.hpp boost/shared_ptr.hpp boost/bind.hpp boost/signal.hpp boost/ref.hpp boost/graph/adjacency_list.hpp boost/graph/graph_traits.hpp boost/graph/topological_sort.hpp boost/graph/graphviz.hpp boost/graph/breadth_first_search.hpp boost/graph/visitors.hpp boost/graph/transpose_graph.hpp boost/any.hpp"

    _boost_libs_mt="boost_thread-mt boost_signals-mt boost_serialization-mt boost_system-mt boost_program_options-mt"

   _boost_libs="boost_thread boost_signals boost_serialization boost_system boost_program_options"
  
  SCI_TRY_LINK(BOOST,boost library and include headers,$_boost_incl,$BOOST_INCLUDE_DIR,$_boost_libs_mt,$BOOST_LIBRARY_DIR,,,optional)

  if test "$HAVE_BOOST" = "no"; then
     SCI_TRY_LINK(BOOST, boost library (non-mt) and include headers,$_boost_incl,$BOOST_INCLUDE_DIR,$_boost_libs,$BOOST_LIBRARY_DIR,,,optional)
  fi                  


  if test "$HAVE_BOOST" = "yes"; then
     DEF_BOOST="#define HAVE_BOOST 1"              
  fi
  if test "$with_boost" != "/usr"; then
    # If with_boost points to /usr, then we don't need to explicitly point to it...
      INC_BOOST_H="-I$with_boost/include"
      BOOST_LIB_DIR_FLAG="$LDRUN_PREFIX$BOOST_LIB_DIR -L$BOOST_LIB_DIR"
  fi
fi
AC_SUBST(HAVE_BOOST)
AC_SUBST(DEF_BOOST)
AC_SUBST(INC_BOOST_H)
AC_SUBST(BOOST_LIB_FLAG)
AC_SUBST(BOOST_LIB_DIR_FLAG)
AC_LANG([C])
##  --  Handle --enable-wasatch_3p... ------------------------

if test "$wasatch_3p" = "yes"; then

  if test "$HAVE_BOOST" != "yes"; then
      AC_MSG_ERROR( Option --enable-wasatch_3p requires Boost... <--with-boost=DIR> )
  fi
#  if test "$HAVE_HDF5" != "yes"; then
#      AC_MSG_ERROR( Option --enable-wasatch_3p requires HDF5... <--with-hdf5=DIR> )
#  fi
  if test "$with_radprops"; then
     AC_MSG_ERROR( Do NOT specify both --with-radprops and --enable-wasatch_3p )
  fi
  if test "$with_tabprops"; then
     AC_MSG_ERROR( Do NOT specify both --with-tabprops and --enable-wasatch_3p )
  fi
  if test "$with_spatialops"; then
     AC_MSG_ERROR( Do NOT specify both --with-spatialops and --enable-wasatch_3p )
  fi
  if test "$with_exprlib"; then
     AC_MSG_ERROR( Do NOT specify both --with-exprlib and --enable-wasatch_3p )
  fi

  mkdir -p Wasatch3P

  $path_to_src/build_scripts/build_wasatch_3p.sh `$sci_pwd` "$with_boost" "$debug"

  if test $? != 0; then
     AC_MSG_ERROR( Failed to build Wasatch 3p libraries... )
  fi

  with_spatialops=`$sci_pwd`/Wasatch3P/install/SpatialOps
  with_exprlib=`$sci_pwd`/Wasatch3P/install/ExprLib
  with_tabprops=`$sci_pwd`/Wasatch3P/install/TabProps
  with_radprops=`$sci_pwd`/Wasatch3P/install/RadProps
fi

##  --  search for SpatialOps Library ------------------------

if test "$with_spatialops"; then
  if test "$HAVE_BOOST" != "yes"; then
    AC_MSG_ERROR( --with-spatialops=DIR requires Boost. )
  fi

  if test "$with_spatialops" = "yes"; then
    AC_MSG_ERROR( --with-spatialops=DIR requires a DIRectory be provided. )
  else

    AC_MSG_CHECKING(for SpatialOps files (For now we are just verifying the existence of lib and .h files)) 
    if test ! -d $with_spatialops; then
      echo ""
      AC_MSG_ERROR( '$with_spatialops' directory does not exist. )
    fi
    SPOPS_INCLUDE_DIR="include/spatialops"
    # Check includes
    for file in FieldExpressions.h FieldFunctions.h OperatorDatabase.h \
                structured/SpatialField.h SpatialOpsConfigure.h SpatialOpsDefs.h SpatialOpsTools.h \
                particles structured; do
      if test ! -e $with_spatialops/$SPOPS_INCLUDE_DIR/$file; then
        echo ""
        AC_MSG_ERROR( '$with_spatialops/$SPOPS_INCLUDE_DIR/$file' does not exist. )
      fi
    done
    # Check lib
    if test ! -f $with_spatialops/lib/spatialops/libspatialops.a; then
      echo ""
      AC_MSG_ERROR( '$with_spatialops/lib/spatialops/libspatialops.a' does not exist. )
    fi
  fi
  AC_MSG_RESULT(yes)
  DEF_SPATIALOPS="#define HAVE_SPATIALOPS 1"
  INC_SPATIALOPS_H="-I$with_spatialops/include"
  SPATIALOPS_LIB_DIR_FLAG="-L$with_spatialops/lib/spatialops"
  SPATIALOPS_LIB_FLAG="-lspatialops -lspatialops-stencil"
  HAVE_SPATIALOPS="yes"
fi
AC_SUBST(HAVE_SPATIALOPS)
AC_SUBST(DEF_SPATIALOPS)
AC_SUBST(INC_SPATIALOPS_H)
AC_SUBST(SPATIALOPS_LIB_FLAG)
AC_SUBST(SPATIALOPS_LIB_DIR_FLAG)

##  --  search for Exprlib Library ------------------------

if test "$with_exprlib"; then
  if test "$HAVE_SPATIALOPS" != "yes"; then
    AC_MSG_ERROR( --with-exprlib=DIR requires SpatialOps. )
  fi

  if test "$with_exprlib" = "yes"; then
    AC_MSG_ERROR( --with-exprlib=DIR requires a DIRectory be provided. )
  else

    AC_MSG_CHECKING(for Exprlib files (For now we are just verifying the existence of lib and .h files)) 
    if test ! -d $with_exprlib; then
      echo ""
      AC_MSG_ERROR( '$with_exprlib' directory does not exist. )
    fi
    SPOPS_INCLUDE_DIR="include/expression"
    # Check includes
    for file in ExprDeps.h ExprLib.h Context.h Expression.h ExpressionID.h FieldManagerList.h \
                ExpressionBase.h ExpressionFactory.h ExpressionTree.h FieldDeps.h FieldManager.h FieldManagerBase.h \
                Functions.h PlaceHolderExpr.h Tag.h TransportEquation.h uintah; do
      if test ! -e $with_exprlib/$SPOPS_INCLUDE_DIR/$file; then
        echo ""
        AC_MSG_ERROR( '$with_exprlib/$SPOPS_INCLUDE_DIR/$file' does not exist. )
      fi
    done
    # Check lib
    if test ! -f $with_exprlib/lib/expression/libexprlib.a; then
      echo ""
      AC_MSG_ERROR( '$with_exprlib/lib/expression/libexprlib.a' does not exist. )
    fi
  fi
  AC_MSG_RESULT(yes)
  DEF_EXPRLIB="#define HAVE_EXPRLIB 1"
  INC_EXPRLIB_H="-I$with_exprlib/include"
  EXPRLIB_LIB_DIR_FLAG="-L$with_exprlib/lib/expression"
  EXPRLIB_LIB_FLAG="-lexprlib"
  HAVE_EXPRLIB="yes"
fi
AC_SUBST(HAVE_EXPRLIB)
AC_SUBST(DEF_EXPRLIB)
AC_SUBST(INC_EXPRLIB_H)
AC_SUBST(EXPRLIB_LIB_FLAG)
AC_SUBST(EXPRLIB_LIB_DIR_FLAG)

##  --  search for TabProps ------------------------

if test "$with_tabprops"; then

  if test "$with_tabprops" = "yes"; then
    AC_MSG_ERROR( --with-tabprops=DIR requires a DIRectory be provided. )
  else
    AC_MSG_CHECKING(for TabProps files (For now we are just verifying the existence of lib and .h files)) 
    if test ! -d $with_tabprops; then
      AC_MSG_ERROR( '$with_tabprops' directory does not exist. )
    fi
    for file in include/tabprops/StateTable.h lib/tabprops/libtabprops.a; do
      if test ! -f $with_tabprops/$file; then
        AC_MSG_ERROR( '$with_tabprops/$file' does not exist. )
      fi
    done
  fi
  AC_MSG_RESULT(yes)
  DEF_TABPROPS="#define HAVE_TABPROPS 1"
  INC_TABPROPS_H="-I$with_tabprops/include"
  TABPROPS_LIB_DIR_FLAG="-L$with_tabprops/lib/tabprops"
  TABPROPS_LIB_FLAG="-ltabprops"
  HAVE_TABPROPS="yes"
fi
AC_SUBST(HAVE_TABPROPS)
AC_SUBST(DEF_TABPROPS)
AC_SUBST(INC_TABPROPS_H)
AC_SUBST(TABPROPS_LIB_FLAG)
AC_SUBST(TABPROPS_LIB_DIR_FLAG)

##  --  search for RadProps ------------------------

if test "$with_radprops"; then

  if test "$with_radprops" = "yes"; then
    AC_MSG_ERROR( --with-radprops=DIR requires a DIRectory be provided. )
  else
    AC_MSG_CHECKING(for RadProps files (For now we are just verifying the existence of lib and .h files)) 
    if test ! -d $with_radprops; then
      AC_MSG_ERROR( '$with_radprops' directory does not exist. )
    fi
    for file in include/radprops/AbsCoeffGas.h include/radprops/Particles.h include/radprops/RadiativeSpecies.h lib/radprops/libradprops.a; do
      if test ! -f $with_radprops/$file; then
        AC_MSG_ERROR( '$with_radprops/$file' does not exist. )
      fi
    done
  fi
  AC_MSG_RESULT(yes)
  DEF_RADPROPS="#define HAVE_RADPROPS 1"
  INC_RADPROPS_H="-I$with_radprops/include"
  RADPROPS_LIB_DIR_FLAG="-L$with_radprops/lib/radprops"
  RADPROPS_LIB_FLAG="-lradprops"
  HAVE_RADPROPS="yes"
fi
AC_SUBST(HAVE_RADPROPS)
AC_SUBST(DEF_RADPROPS)
AC_SUBST(INC_RADPROPS_H)
AC_SUBST(RADPROPS_LIB_FLAG)
AC_SUBST(RADPROPS_LIB_DIR_FLAG)

##  --  Handle --enable-wasatch-in-arches --------------------------------

if test "$wasatch_in_arches" = "yes"; then
   if test "$USE_WASATCH" != yes; then
      add_important_warning "Turning on Wasatch for use in Arches."
      AC_MSG_WARN(Turning on Wasatch for use in Arches.)
      USE_WASATCH=yes
   fi
   DEF_WASATCH="#define WASATCH_IN_ARCHES 1";
   BUILD_WASATCH_IN_ARCHES="yes"
fi
AC_SUBST(DEF_WASATCH_IN_ARCHES)
AC_SUBST(BUILD_WASATCH_IN_ARCHES)

##  --  Determine whether we can use WASATCH OR NOT ----------------------

if test "$USE_WASATCH" = "yes"; then
   if test "$HAVE_BOOST" = "yes" &&
     test "$HAVE_TABPROPS" = "yes" &&
     test "$HAVE_SPATIALOPS" = "yes" &&
     test "$HAVE_EXPRLIB" = "yes"; then
    BUILD_WASATCH=yes
  else
    AC_MSG_ERROR(You specified the use of the Wasatch component but do not have all the of the prereqs: Boost ($HAVE_BOOST); TabProps ($HAVE_TABPROPS); SpatialOps ($HAVE_SPATIALOPS); and ExprLib ($HAVE_EXPRLIB).)
  fi
else
  if test "$HAVE_BOOST" = "yes" &&
     test "$HAVE_TABPROPS" = "yes" &&
     test "$HAVE_SPATIALOPS" = "yes" &&
     test "$HAVE_EXPRLIB" = "yes"; then
    BUILD_WASATCH=yes
    add_important_warning "All prereqs satisfied, turning on Wasatch component."
    AC_MSG_WARN(All prereqs satisfied, turning on Wasatch component.)
  fi
fi

if test "$BUILD_WASATCH" = no; then 
  DEF_WASATCH="#define NO_WASATCH";
fi
AC_SUBST(BUILD_WASATCH)

##  --  search for hypre stuff  ------------------------------------------

_sci_lib=
_sci_inc=
_sci_optional=optional

# Handle --without option
if test "$with_hypre" = "no"; then
   AC_MSG_WARN( "Hypre is turned off." )
else
   if test "$with_hypre" = "yes"; then
      _sci_optional=required
   elif test "$with_hypre"; then
      _sci_lib="$with_hypre/lib"
      _sci_inc="$with_hypre/include"
      _sci_optional=specific
   fi

   # # We used to also check for HYPRE_blas, but hypre 1.9 doesn't have it,
   # # so we aren't going to check for it in any version.  If the other 3
   # # libs exist, we will assume that HYPRE_blas is there if it is needed.
   # SCI_TRY_LINK(HYPRE,HYPRE libraries and headers,HYPRE.h,$_sci_inc,
   #                          HYPRE_struct_mv HYPRE_struct_ls HYPRE_sstruct_mv HYPRE_sstruct_ls HYPRE_LSI HYPRE_superlu HYPRE_utilities,
   #                          $_sci_lib,-lm,,$_sci_optional)

   # We should just be able to link vs. libHYPRE.so...
   #   Also, libHYPRE has some C++ stuff in it, so we need to be in C++ mode...
   AC_LANG([C++])
   SCI_TRY_LINK(HYPRE,HYPRE libraries and headers,HYPRE.h,$_sci_inc,
                            HYPRE,
                            $_sci_lib,
                            $MPI_LIB_DIR_FLAG $MPI_LIB_FLAG $MALLOC_TRACE_LIB_DIR_FLAG $MALLOC_TRACE_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG $M_LIB_FLAG $DL_LIB_FLAG $THREAD_LIB_FLAG,
                            ,
                            $_sci_optional)
   AC_LANG([C])

   if test "$HAVE_HYPRE" = "yes" ; then
      DEF_HYPRE="#define HAVE_HYPRE 1"
   fi

   if test "$with_hypre" && test ! "$DEF_HYPRE"; then
      echo
      AC_MSG_ERROR(One or more of the HYPRE components is missing.)
   fi
fi

AC_SUBST(HAVE_HYPRE)
AC_SUBST(DEF_HYPRE)
AC_SUBST(INC_HYPRE_H)
AC_SUBST(HYPRE_LIB_FLAG)
AC_SUBST(HYPRE_LIB_DIR_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/hypre_testdefs.h"


##  --  search for cuda stuff  ------------------------------------------
if test "$sci_required_cuda" = "yes"; then

  echo
  AS_MESSAGE(checking for required CUDA components......)
  echo
  
  # This is now turned off to test some static builds on the GPU-enabled Jaguar-XK6
  # guard against building staticaly linked Uintah with CUDA
  #if test \( "$with_cuda" \) -a \( "$static" != "no" -o "$SO_OR_ARCHIVE_EXTENSION" = "a" \); then
    #AC_MSG_ERROR( [Cannot build a statically linked version of Uintah with CUDA support...] )
  #fi

  # check that the CUDA installation and SDK exist
  if ! test -d "$with_cuda"; then
    AC_MSG_ERROR( [The directory specified with --with-cuda=DIR is not valid: $with_cuda] )
  fi

  # the major CUDA definitions, etc
  DEF_CUDA=""
  HAVE_CUDA=""
  HAVE_CUDA_H=""
  INC_CUDA_H=""
  CUDA_LIB_DIR=""
  NVCCEXT=""
  NVCC_MODULE_FLAGS=""
  DEF_PTXDIR_ABS=""

  # all the -L's
  CUDA_LIB_DIR_FLAG=""

  # all the -l's
  CUDA_LIB_FLAG=""

  # confirm the existence/location of CUDA lib and include directories
  _cuda_lib=
  _cuda_inc="$with_cuda/include"
  # account for lib64 or lib32 or just lib...
  BASE_LIB_PATH(_cuda_lib,$with_cuda)
  
# code to compile and test various system GPU/CUDA properties
# (these files, etc are deleted after tests)
#------------------------------------------------------------------------------
# test CUDA and CUDA_SDK libraries and headers
  cat > cuda_conf_test.cu << EOF
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

__global__ void vectorAddGPU(float *a, float *results, int N) {
	int idx = threadIdx.x;
	if (idx < N) {
		*(results+idx) = *(a+idx) * (*(a+idx));
	}
}

int main(int argc, char **argv) {

	int N = 512;
  float *a = (float *)malloc(N * sizeof(float));
  float *results = (float *)malloc(N * sizeof(float));
	float *dev_a, *dev_results;
	int memsize = N * sizeof(float);

	for (int i = 0; i < N; i++) {
	  *(a+i) = i;
		printf("a:%d = %f\n", i, *(a+i));
	}
	
	cudaMalloc((void**)&dev_a, memsize);
	cudaMalloc((void**)&dev_results, memsize);
	cudaMemcpy(dev_a, a, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_results, results, memsize, cudaMemcpyHostToDevice);

	vectorAddGPU<<< 1, 512 >>>(dev_a, dev_results, N);

	cudaMemcpy(results, dev_results, memsize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("a:%d = %f\n", i, *(results+i));
	}

  cudaFree(dev_a);
  cudaFree(dev_results);
  free(a);
  free(results);

	cudaDeviceReset();

	return 1;
}
EOF

  SCI_CUDA_COMPILE_LINK_TEST(CUDA,
                             CUDA libraries and headers,
                             cuda.h,
                             $_cuda_inc,
                             cudart cuda,
                             $_cuda_lib,
                             ,
                             cuda_conf_test.cu,
                             optional,
                             yes)

rm -rf cuda_conf_test*
echo

#[aph] - debug
#echo
#echo "HAVE_CUDA         = " $HAVE_CUDA
#echo "HAVE_CUDA_H       = " $HAVE_CUDA_H
#echo "INC_CUDA_H        = " $INC_CUDA_H
#echo "CUDA_LIB_FLAG     = " $CUDA_LIB_FLAG
#echo "CUDA_LIB_DIR      = " $CUDA_LIB_DIR
#echo "CUDA_LIB_DIR_FLAG = " $CUDA_LIB_DIR_FLAG
#echo "NVCC_CFLAGS       = " $NVCC_CFLAGS
#echo "NVCC_CXXFLAGS     = " $NVCC_CXXFLAGS
#echo "NVCC_LDFLAGS      = " $NVCC_LDFLAGS
#echo "NVCC_LIBS         = " $NVCC_LIBS
#echo "CC                = " $CC
#echo "NVCC              = " $NVCC
#echo "LIBS              = " $LIBS
#echo "LDFLAGS           = " $LDFLAGS
#echo "CFLAGS            = " $CFLAGS
#echo "CXXFLAGS          = " $CXXFLAGS
#echo "NVCC_MODULE_FLAGS = " $NVCC_MODULE_FLAGS
#echo "DEF_PTXDIR_ABS    = " $DEF_PTXDIR_ABS
#echo

fi #end sci_required_cuda

# these are all defined after the call to SCI_CUDA_COMPILE_LINK_TEST
AC_SUBST(HAVE_CUDA)
AC_SUBST(DEF_CUDA)
AC_SUBST(INC_CUDA_H)
AC_SUBST(CUDA_LIB_DIR_FLAG)
AC_SUBST(CUDA_LIB_FLAG)
AC_SUBST(NVCC_CFLAGS)
AC_SUBST(NVCC_CXXFLAGS)
AC_SUBST(NVCC_LIBS)
AC_SUBST(NVCCEXT)
AC_SUBST(NVCC_MODULE_FLAGS)
AC_SUBST(DEF_PTXDIR_ABS)

sci_defs_files="$sci_defs_files include/sci_defs/cuda_testdefs.h"


##  --  finis flags  ------------------------------------------------
case $host in
  *linux*)
    ;;
  *aix*)
    ;;
  *dec*)
    ;;
  *)
    ;;
esac


##  ---------------------------------------------------------------------
##  --------------------  Fix Frost config vars  ------------------------
##  ---------------------------------------------------------------------
##  Now that we have run configure with the (xlc,xlC,xlf) compilers, we
##  need to change those to (mpxlc,mpxlC,mpxlf) which automatically use
##  the mpi libraries, so we need to set the MPI variables to be blank.
##  ---------------------------------------------------------------------

  case $host in
    *aix*)
        AC_MSG_WARN(Changing the compiler for AIX computers...)
        AC_MSG_WARN(...to (probably) mpxlC, etc, and removing MPI flags...)
	CC=`echo $CC | sed 's,x,mpx,g'`
	CXX=`echo $CXX | sed 's,x,mpx,g'`
	F77=`echo $F77 | sed 's,x,mpx,g'`
	MPI_LIB_FLAG=""
	MPI_LIB_DIR_FLAG=""
	INC_MPI_H=""
	;;
  esac


##  ---------------------------------------------------------------------
##  --------------------  generate output files  ------------------------
##  ---------------------------------------------------------------------

if test -n "$WOFF_NUMBERS"; then
  for num in $WOFF_NUMBERS; do
    if test -z "$WOFF_COMMA_LIST"; then
       WOFF_COMMA_LIST="$num"
    else
       WOFF_COMMA_LIST="$WOFF_COMMA_LIST,$num"
    fi
    WOFF_STRING="$WOFF_STRING -Wl,-woff -Wl,$num"
  done
  WOFF_STRING="-woff $WOFF_COMMA_LIST $WOFF_STRING"
fi
AC_SUBST(WOFF_STRING)

SCI_ARG_VAR(CXXFLAGS,C++ compiler flags)

scirun_obj_dir=`pwd`;
scirun_src_dir=`(cd ${srcdir};pwd)`

if test "$IS_WIN" = "yes"; then
  scirun_obj_dir=`echo "$scirun_obj_dir" | sed 's,/,,' | sed 's,/,:/,'`
  scirun_src_dir=`echo "$scirun_src_dir" | sed 's,/,,' | sed 's,/,:/,'`
fi

AC_SUBST(scirun_obj_dir)
AC_SUBST(scirun_src_dir)

AC_CONFIG_FILES(configVars.mk Makefile disjointPackageMakefile $sci_defs_files)

##  Add a list of Uintah specific output files.
AC_CONFIG_FILES(tools/fspec.pl)

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


for testfile in $sci_defs_files; do
  file=`echo $testfile | sed 's/test//'`

  if cmp -s $file $testfile 2>/dev/null; then
    AC_MSG_RESULT($file is unchanged.)
    rm -f $testfile
  else
    mv $testfile $file
    AC_MSG_WARN($file has changed.)

    # Because the make system isn't smart enough to notice when a new
    # petsc or hypre is specified to configure, these checks help clean up the code.
    # The "-d StandAlone" is used to skip these checks if this is the first time
    # we have run configure.

    if test -d StandAlone -a "$file" = "include/sci_defs/petsc_defs.h"; then
       echo "WARNING: petsc dependent .o files are probably out of date... removing..."
       $path_to_src/build_scripts/repair.sh -quiet petsc
    fi

    if test -d StandAlone -a "$file" = "include/sci_defs/hypre_defs.h"; then
       echo "WARNING: hypre dependent .o files are probably out of date... removing..."
       $path_to_src/build_scripts/repair.sh -quiet hypre
    fi

  fi
done

###############
## Re-display any IMPORTANT_WARNINGS that have been collected during configure in hopes
## that the user actually sees them.

if (( $NUM_IMPORTANT_WARNINGS > 0 )); then

   echo ""
   echo "!!! IMPORTANT WARNINGS - Please Read: !!!"
   echo ""
   echo "NOTE: Some IMPORTANT WARNINGS were generated during configuration.  A summary of"
   echo "these messages can be found below:"
   echo

   # Note: need '[[' below to get autoconf to create a single '[' in the actual configure file.
   for (( i=0; i < $NUM_IMPORTANT_WARNINGS; i++ )); do
      echo " - " ${IMPORTANT_WARNINGS[[${i}]]}
   done
   echo
fi

#### Clean up: ####
## Possible temporary template directories
rm -rf ii_files
## sedscript
rm -f sedscript
