##
## The MIT License
##
## Copyright (c) 1997-2017 The University of Utah
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
##  PLEASE DO NOT COMMIT ANY CHANGES WITH OUT HAVING ONE OF THE 
##  ABOVE REVIEW YOUR CHANGES.
##
##  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##  --------------------------------------------------------------------
##  ------------------------  initialize  ------------------------------
##  --------------------------------------------------------------------

##  initialize configure 
AC_INIT([Uintah],[2.0],[uintah-developer@gforge.sci.utah.edu])

##  require autoconf version 2.69
##     The reason to require the latest autoconf is to try and get everyone who
##     updates configure to use the same version.  By doing so, the SVN diffs
##     don't look completely different from minor changes.  (This happens
##     if people switch back and forth between different versions of autoconf.)
##
##     type "autoconf --version" to find out what version you have.
##
AC_PREREQ(2.69)

AC_COPYRIGHT(
##
## The MIT License
##
## Copyright (c) 1997-2017 The University of Utah
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
   AC_MSG_WARN(___ eg: ../sgi32opt or ../linux64dbg)
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

## 
## IMPORTANT!!!  Configure developers, PLEASE READ:
##
## The following BULLET PROOFING needs to be placed manually into
## configure every time after autoconf is run, and before configure is
## checked in:
##
## To avoid allowing invalid arguments to configure, we have to manually
## hack the 'configure' script after it has been 'autoconf'd, and add the
## line "enable_option_checking=fatal".  There apparently is no way to do
## this through autoconf macros... sigh.  Manually find this entire text
## block (comment and all) in configure, MOVE IT ALL TO THE TOP OF THE FILE
## (just after the MIT license), and uncomment the following line:
##
# enable_option_checking=fatal

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
   AC_MSG_WARN(Using unstable version of pwd)
   sci_pwd=pwd
fi

##  Initialize the recognized argument lists so that configure can
##  verify that all args provided are valid (ie: avoid misspelled
##  args from slipping in).
##
##  NOTE: sci_arg_with_list does not appear to actually be used
##  anymore to verify configure arguments... sigh.
##
sci_arg_with_list="--no-create --no-recursion"
sci_arg_enable_list=""

##  --------------------------------------------------------------------
##  ------------------  optional variable intialization  ---------------
##  --------------------------------------------------------------------

# Setting variables to 'no' isn't required, but is useful in displaying
# some error condition information to the user.
HAVE_BOOST=no
HAVE_SPATIALOPS=no
HAVE_EXPRLIB=no
HAVE_TABPROPS=no
HAVE_RADPROPS=no
HAVE_NSCBC=no
HAVE_POKITT=no

##  --------------------------------------------------------------------
##  ------------------  check for all arguments     --------------------
##  --------------------------------------------------------------------

##
##  Use the SCI_ARG_WITH and SCI_ARG_ENABLE macros to check for arguments
##

SCI_ARG_WITH(atlas,
  AC_HELP_STRING(--with-atlas=DIR,
                 Use the atlas installation (for lapack/blas) in DIR),
  [sci_required_atlas=yes],
  [with_atlas=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(math,
  AC_HELP_STRING(--with-math=FULL_PATH_INCLUDING_LIB,
                 Use specified math library (Defaults to built in -lm)),
  [with_math="$withval"],
  [with_math=NOT_SET],
  FILE,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(mkl,
  AC_HELP_STRING(--with-mkl=DIR,
                 Use the mkl (for lapack/blas) installation in DIR),
  [with_mkl="$withval"
   sci_required_mkl=yes],
  [with_mkl=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(essl,
  AC_HELP_STRING(--with-essl=DIR,
                 Use the essl (for lapack/blas) installation in DIR),
  [with_essl="$withval"
   sci_required_essl=yes],
  [with_essl=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(acml,
  AC_HELP_STRING(--with-acml=DIR,
                 Use the acml (for lapack/blas) installation in DIR),
  [with_acml="$withval"
   sci_required_acml=yes],
  [with_acml=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(blas,
  AC_HELP_STRING(--with-blas=DIR,
                 Use the blas installation in DIR),
  [with_blas="$withval"
   sci_required_blas=yes],
  [with_blas=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(lapack,
  AC_HELP_STRING(--with-lapack=DIR, 
                 Use the lapack installation in DIR),
  [with_lapack="$withval"
   sci_required_lapack=yes],
  [with_lapack=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(fortran,
  AC_HELP_STRING(--with-fortran=DIR, 
                 Use the fortran (eg: g2c) installation in DIR/lib),
  [with_fortran="$withval"
   sci_required_fortran=yes],
  [with_fortran=NOT_SET],
  DIR,
  WITHOUT-IS-VALID)
  
SCI_ARG_WITH(ofed,
  AC_HELP_STRING(--with-ofed=DIR,
                 Specify that OFED - Infiniband - is needed for the mpi installation),
  [with_ofed="$withval"
   sci_required_ofed=no],
  [with_ofed=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(mx,
  AC_HELP_STRING(--with-mx=DIR,
                 Specify that mx - Myrinet - is needed for the mpi installation),
  [with_mx="$withval"
   sci_required_mx=no],
  [with_mx=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(pokitt,
  AC_HELP_STRING(--with-pokitt=DIR,
                 Use the PoKiTT installation in DIR),
  [with_pokitt="$withval"
   sci_required_pokitt=yes],
  [with_pokitt=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(mpi,
  AC_HELP_STRING(--with-mpi=DIR,
                 Use the mpi installation in DIR. Set to 'built-in' if  mpi is automatically handled by your compiler.),
  [with_mpi="$withval"],
  [with_mpi=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(mpi_lib,
  AC_HELP_STRING(--with-mpi-lib=DIR,
                 Use the mpi libraries installed in DIR),
  [with_mpi_lib="$withval"
   sci_required_mpi=yes],
  [with_mpi_lib=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(mpi_include,
  AC_HELP_STRING(--with-mpi-include=DIR,
                 Use the mpi include files installed in DIR),
  [with_mpi_include="$withval"
   sci_required_mpi=yes],
  [with_mpi_include=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_WITH(mpi_extra_libs,
  AC_HELP_STRING(--with-mpi-extra-libs=/path/to/lib1.so,
                 Use this additional library (or libraries - comma separated list) for mpi linking.),
  [with_mpi_extra_libs="$withval"],
  [with_mpi_extra_libs=NOT_SET],
  FILES,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(cuda,
  AC_HELP_STRING(--with-cuda=DIR,
                 Use the CUDA installation in DIR),
  [with_cuda="$withval"
   sci_required_cuda=yes],
  [with_cuda=NOT_SET
   sci_required_cuda=no],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(hypre,
  AC_HELP_STRING(--with-hypre=DIR,
                 Use the hypre installation in DIR.  Set to 'built-in' if explicit linking of hypre is NOT required. ),
  [with_hypre="$withval"
   sci_required_hypre=yes],
  [with_hypre=NOT_SET],
  DIR,
  WITHOUT-IS-VALID)
  
SCI_ARG_WITH(hypre_lib,
  AC_HELP_STRING(--with-hypre-lib=DIR,
                 Use the hypre libraries installed in DIR),
  [with_hypre_lib="$withval"
   sci_required_hypre=yes],
  [with_hypre_lib=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_WITH(hypre_include,
  AC_HELP_STRING(--with-hypre-include=DIR,
                 Use the hypre include files installed in DIR),
  [with_hypre_include="$withval"
   sci_required_hypre=yes],
  [with_hypre_include=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_WITH(petsc,
  AC_HELP_STRING(--with-petsc=DIR,
                 Use the petsc installation in DIR (or set to 'built-in'). ),
  [with_petsc="$withval"
   sci_required_petsc=yes],
  [with_petsc=NOT_SET],
  DIR,
  WITHOUT-IS-VALID)
  
SCI_ARG_WITH(boost,
  AC_HELP_STRING(--with-boost=DIR,
                 Use the Boost library in DIR/lib),
  [with_boost="$withval"
   sci_required_boost=yes],
  [with_boost=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(boost_library,
  AC_HELP_STRING(--with-boost-library=DIR,
                 Use the Boost library in DIR),
  [with_boost_library="$withval"
   sci_required_boost=yes],
  [with_boost_library=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(boost_include,
  AC_HELP_STRING(--with-boost-include=DIR,
                 Use the Boost include files in DIR),
  [with_boost_include="$withval"
   sci_required_boost=yes],
  [with_boost_include=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(spatialops,
  AC_HELP_STRING(--with-spatialops=DIR,
                 Use the Wasatch 3P SpatialOps library in DIR),
  [with_spatialops="$withval"],
  [with_spatialops=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(exprlib,
  AC_HELP_STRING(--with-exprlib=DIR,
                 Use the Wasatch 3P ExprLib library in DIR),
  [with_exprlib="$withval"],
  [with_exprlib=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(tabprops,
  AC_HELP_STRING(--with-tabprops=DIR,
                 Use the Wasatch 3P TabProps library in DIR),
  [with_tabprops="$withval"],
  [with_tabprops=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(radprops,
  AC_HELP_STRING(--with-radprops=DIR,
                 Use the Wasatch 3P RadProps library in DIR),
  [with_radprops="$withval"],
  [with_radprops=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(nscbc,
  AC_HELP_STRING(--with-nscbc=DIR,
                 Use the Wasatch 3P NSCBC library in DIR),
  [with_nscbc="$withval"],
  [with_nscbc=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
# This specification must be after the individual W3P flags,
# and is used as a shortcut so you don't have to specify all 5 individual flags...
SCI_ARG_WITH(wasatch3p,
  AC_HELP_STRING(--with-wasatch3p=DIR,
                 Use the SpatialOps/ExprLib/TabPros/RadProps/NSCBC libraries in DIR),
  [with_wasatch3p="$withval"],
  [with_wasatch3p=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_WITH(zoltan,
  AC_HELP_STRING(--with-zoltan=DIR,
                 Use the Zoltan installation in DIR),
  [with_zoltan="$withval"
   sci_required_zoltan=yes],
  [with_zoltan=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(papi,
  AC_HELP_STRING(--with-papi=DIR,
                 Use the papi installation in DIR),
  [with_papi="$withval"],
  [with_papi=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(gperftools,
  AC_HELP_STRING(--with-gperftools=DIR,
                 Use the Google Perftools installation in DIR),
  [with_gperftools="$withval"],
  [with_gperftools=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_WITH(ipm,
  AC_HELP_STRING(--with-ipm=DIR,
                 Use the ipm installation in DIR),
  [with_ipm="$withval"],
  [with_ipm=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(magma,
  AC_HELP_STRING(--with-magma=DIR,
                 Use the MAGMA installation in DIR),
  [with_magma="$withval"
   sci_required_magma=yes],
  [with_magma=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(zlib,
  AC_HELP_STRING(--with-zlib=DIR,
                 Use the zlib installation in DIR - checks that $DIR/lib[[64]] and $DIR/include exist),
  [with_zlib="$withval"
   sci_required_zlib=yes],
  [with_zlib=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(jpeg,
  AC_HELP_STRING(--with-jpeg=DIR,
                 Use the jpeg installation in DIR),
  [with_jpeg="$withval"
   sci_required_jpeg=yes],
  [with_jpeg=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(tiff,
  AC_HELP_STRING(--with-tiff=DIR,
                 Use the tiff installation in DIR),
  [with_tiff="$withval"
   sci_required_tiff=yes],
  [with_tiff=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_WITH(xalan,
  AC_HELP_STRING(--with-xalan=DIR,
                 Use the xalan installation in DIR),
  [with_xalan="$withval"
   sci_required_xalan=yes],
  [with_xalan=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_WITH(libxml2,
  AC_HELP_STRING(--with-libxml2=DIR,
                 Use the libxml2 installation in DIR.  Set 'DIR' to 'built-in' if explicit linking of xml2 is NOT required.),
  [with_libxml2="$withval"
   sci_required_libxml2=yes],
  [with_libxml2=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(visit,
  AC_HELP_STRING(--with-visit=DIR,
                 Use the visit installation in DIR),
  [with_visit="$withval"
   sci_required_visit=yes],
  [with_visit=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)
  
SCI_ARG_WITH(visus,
  AC_HELP_STRING(--with-visus=DIR,
                 Turn on ViSUS IDX support.  Point to ViSUS installation.),
  [with_visus="$withval"],
  [with_visus=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_WITH(pidx,
  AC_HELP_STRING(--with-pidx=DIR,
                 Turn on ViSUS Parallel IDX support.  Point to PIDX installation.),
  [with_pidx="$withval"],
  [with_pidx=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_WITH(MallocTrace,
  AC_HELP_STRING(--with-MallocTrace=DIR,
                 Use MallocTrace library.  Note: you cannot use sci-malloc at the same time.),
  [with_MallocTrace="$withval"],
  [with_MallocTrace=NOT_SET],
  DIR,
  WITHOUT-IS-NOT-VALID)

SCI_ARG_ENABLE(wasatch_3p,
  AC_HELP_STRING(--enable-wasatch_3p,
                 Download and build the Wasatch thirdparty libraries - TabProps ExprLibs SpatialOps RadProps NSCBC.),
  [build_wasatch_3p="$enableval"],
  [build_wasatch_3p="no"])

SCI_ARG_ENABLE(pidx,
  AC_HELP_STRING(--enable-pidx,
                 Download and build the PIDX thirdparty library),
  [build_pidx_lib="$enableval"],
  [build_pidx_lib="no"])

SCI_ARG_ENABLE(static,
  AC_HELP_STRING(--enable-static,
                 Build a statically linked version of Uintah.  Set to 'full' to force full static link.),
  [static="$enableval"],
  [static="no"])
  
SCI_ARG_ENABLE(verbose,
  AC_HELP_STRING(--enable-verbose,
                 Turn on verbose configure output),
  [verbose="yes"],
  [verbose="no"])
  
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
  AC_HELP_STRING(--enable-ray-scatter,
                 Turn on ray scattering in Arches),
  [ray_scatter="$enableval"],
  [ray_scatter=${ray_scatter=''}])
  
SCI_ARG_ENABLE(gencode,
  AC_HELP_STRING(--enable-gencode,
                 Specify compute capability for GPU code generation (allowable values: 20, 21, 30, 35)),
  [cuda_gencode="$enableval"],
  [cuda_gencode=${gencode='30'}])
  
SCI_ARG_ENABLE(largesos,
  AC_HELP_STRING(--enable-largesos,
                 Build one .so per directory),
  [largesos="$enableval"],
  [largesos=${largesos='no'}])
  
SCI_ARG_ENABLE(sci-malloc,
  AC_HELP_STRING(--enable-sci-malloc,
                 Use SCI memory allocation/deallocation checks.),
  [enable_sci_malloc="$enableval"],
  [enable_sci_malloc=${enable_sci_malloc='no'}])
  
SCI_ARG_ENABLE(memory-init,
  AC_HELP_STRING(--enable-memory-initialization=8_bit_number,
                 Force all new'd and malloc'd memory to be initialied to the value given - defaults to 127.),
  [enable_memory_initialization="$enableval"],
  [enable_memory_initialization=${enable_memory_initialization='no'}])
  
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

SCI_ARG_ENABLE(rpath,
  AC_HELP_STRING(--enable-rpath,
                 Set a default runtime library search path),
  [enable_rpath="$enableval"],
  [enable_rpath=${enable_rpath='yes'}])

#
# ---- List of --enable statements for Uintah Components ---- 
#

SCI_ARG_ENABLE(arches,
  AC_HELP_STRING(--enable-arches,
                 Turn on the Arches component - and Radiation Model.),
  [BUILD_ARCHES="yes"],
  [BUILD_ARCHES="no"
   DEF_ARCHES=["#define NO_ARCHES"]])

SCI_ARG_ENABLE(fvm,
  AC_HELP_STRING(--enable-fvm,
                 Turn on the FVM component.),
  [BUILD_FVM="yes"],
  [BUILD_FVM="no"
   DEF_FVM=["#define NO_FVM"]])

SCI_ARG_ENABLE(ice,
  AC_HELP_STRING(--enable-ice,
                 Turn on the Ice component.),
  [BUILD_ICE="yes"],
  [BUILD_ICE="no"
   DEF_ICE=["#define NO_ICE"]])

SCI_ARG_ENABLE(mpm,
  AC_HELP_STRING(--enable-mpm,
                 Turn on the Mpm component.),
  [BUILD_MPM="yes"],
  [BUILD_MPM="no"
   DEF_MPM=["#define NO_MPM"]])

SCI_ARG_ENABLE(wasatch,
  AC_HELP_STRING(--enable-wasatch,
                 Turn on the Wasatch component.),
  [BUILD_WASATCH="yes"],
  [BUILD_WASATCH="no"
   DEF_WASATCH=["#define NO_WASATCH"]])

SCI_ARG_ENABLE(heat,
  AC_HELP_STRING(--enable-heat,
                 Turn on the Heat component.),
  [BUILD_HEAT="yes"],
  [BUILD_HEAT="no"
   DEF_HEAT=["#define NO_HEAT"]])

SCI_ARG_ENABLE(phasefield,
  AC_HELP_STRING(--enable-phasefield,
                 Turn on the PhaseField component.),
  [BUILD_PHASEFIELD="yes"],
  [BUILD_PHASEFIELD="no"
   DEF_PHASEFIELD=["#define NO_PHASEFIELD"]])

SCI_ARG_ENABLE(all_components,
  AC_HELP_STRING(--enable-all-components,
                 Turn on the all components.),
  [use_all_components="yes"
   BUILD_ARCHES="yes"
   BUILD_FVM="yes"
   BUILD_ICE="yes"
   BUILD_MPM="yes"
   BUILD_WASATCH="yes"
   BUILD_HEAT="yes"
   BUILD_PHASEFIELD="yes"
   DEF_ARCHES=
   DEF_FVM=
   DEF_ICE=
   DEF_MPM=
   DEF_WASATCH=
   DEF_HEAT=
   DEF_PHASEFIELD=
  ],
  [use_all_components="no"])

#
# --- end list of -enable components ------
#

BUILD_MODELS_RADIATION=yes
DEF_MODELS_RADIATION=

if test "$use_all_components" = "yes" -a          \
        \( "$enable_arches"           = "yes" -o  \
           "$enable_fvm"              = "yes" -o  \
           "$enable_ice"              = "yes" -o  \
           "$enable_models_radiation" = "yes" -o  \
           "$enable_mpm"              = "yes" -o  \
           "$enable_wasatch"          = "yes" -o  \
           "$enable_heat"             = "yes" -o  \
           "$enable_phasefield"       = "yes" \); then
   AC_MSG_ERROR( Please do not use both --enable-all-components and --enable-<specific-component> )
fi
if test "$BUILD_ARCHES"     != "yes"           -a   \
        "$BUILD_FVM"        != "yes"           -a   \
        "$BUILD_ICE"        != "yes"           -a   \
        "$BUILD_MPM"        != "yes"           -a   \
        "$BUILD_WASATCH"    != "yes"           -a   \
        "$BUILD_HEAT"       != "yes"           -a   \
        "$BUILD_PHASEFIELD" != "yes" ; then
  add_important_warning "-------------------------------------------------------------------------"
  msg="!!! NO COMPONENTS WERE SPECIFIED! ONLY UTILITY PROGRAMS WILL BE BUILT !!!"
  add_important_warning "$msg"
  add_important_warning "-------------------------------------------------------------------------"
fi

if test "$BUILD_ICE" = "yes" -a "$BUILD_MPM" = "no"; then
   AC_MSG_ERROR( [If you enable ICE, you must enable MPM too.] )
fi

if test "$enable_sci_malloc" = "yes" -a "$with_MallocTrace" != NOT_SET ; then
   AC_MSG_ERROR( You cannot use both --with-MallocTrace and --enable-sci-malloc! )
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
         AC_MSG_ERROR( --enable-memory-initialization's value must be from 0 to 127. )
      fi
   fi
   DEF_MEMORY_INIT_NUMBER="#define MEMORY_INIT_NUMBER $memory_init"
fi

AC_SUBST(DEF_MEMORY_INIT)
AC_SUBST(DEF_MEMORY_INIT_NUMBER)
sci_defs_files="$sci_defs_files Core/Malloc/testmem_init.h"

if test "$enable_rpath" = "no"; then
   AC_MSG_WARN(You have removed the rpath flag from some libraries.)
   AC_MSG_WARN(This may have unpredictable affects.)
   AC_MSG_WARN(Do not do this unless you really know what you are doing.)
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

IS_OSX="no"

unset NUM_CPUS

MAKE_ARCHIVES="no"

case $host in
  *linux*)
    SCI_CHECK_OS_VERSION(2.2.17,,
      configure: error: wrong Linux version. exiting.;exit)
    NUM_CPUS="`cat /proc/cpuinfo | fgrep processor | wc | cut -c1-7`"
    ;;
  *-darwin*)
    NUM_CPUS=2
    IS_OSX="yes"
    DEF_OSX="#define IS_OSX 1"
    ;;
  *)
    echo
    AC_MSG_WARN(!!!!!! This system is untested - You are on your own !!!!!!)
    echo
    NUM_CPUS="1"
    ;;
esac

AC_SUBST(IS_OSX)
AC_SUBST(DEF_OSX)

sci_defs_files="$sci_defs_files include/sci_defs/osx_testdefs.h"

AC_MSG_CHECKING(physical processor count)
if test -z "$NUM_CPUS"; then
  NUM_CPUS="1"
fi
AC_MSG_RESULT($NUM_CPUS)

##############################################################################
#
# Apparently AC_PROG_CC will set CFLAGS to "-O2 -g" if you don't specify
# its value first, so... (And AC_CHECK_SIZEOF apparently calls AC_PROG_CC...)
#
CFLAGS=$CFLAGS

##############################################################################
#
# Determine which compiler we are using.  If necessary, remind the user
# that they did not specify a specific compiler on the configure line.
#

if test "$CC" = ""; then
   AC_MSG_WARN(You did not specify which compiler to use so we will search for the default...)
   CC_NOT_SPECIFIED_BY_USER=yes
fi

AC_PROG_CC([$CC gcc cc])

if test "$CC_NOT_SPECIFIED_BY_USER" = "yes"; then

  AC_MSG_WARN(C compiler not specified with 'CC' environment variable. Using default... $CC located here:)
  # Let the reader know exactly which CC is being used:
  which $CC
fi

# Figure out which compiler we are using (gcc, clang, etc)
AC_MSG_CHECKING(the type of compiler we are using)
USING_GCC=no
USING_ICC=no
USING_CLANG=no
USING_XLC=no
USING_PGCC=no

($CC --version | grep -iq "Free Software Foundation") &> /dev/null
if test "$?" = 0; then
  AC_MSG_RESULT(Using the GCC compiler.)
  USING_GCC=yes
else
  ($CC --version | grep -iq "Intel") &> /dev/null
  if test "$?" = 0; then
    AC_MSG_RESULT(Using the Intel ICC compiler.)
    USING_ICC=yes
  else
    ($CC --version | grep -iq "clang") &> /dev/null
    if test "$?" = 0; then
      AC_MSG_RESULT(Using the Clang compiler.)
      # Found clang compiler... verify that it isn't the Apple version as the Apple version is broken...
      #$CC --version | grep -iq "Apple" 
      #if test "$?" = 0; then
      #  AC_MSG_ERROR([Found clang, but the Apple version which is not supported.  Please use a different compiler.])
      #fi
      USING_CLANG=yes
    else
      ($CC -qversion | grep -iq "IBM XL") &> /dev/null
      if test "$?" = 0; then
        AC_MSG_RESULT(Using the XLC compiler.)
        USING_XLC=yes
      else
        ( $CC -V | grep -iq "pgcc" ) &> /dev/null
        if test "$?" = 0; then
          AC_MSG_RESULT(Using the PGCC compiler.)
          USING_PGCC=yes
        else
          AC_MSG_WARN(Using an unknow type of compiler...)
        fi
      fi
    fi
  fi
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
        ;;
  esac
  AC_MSG_WARN(C++ compiler not specified with 'CXX' environment variable.  Using default... $CXX located here:)
  which $CXX
fi

##  ----------  determine if this is a 32 or 64 bit machine

AC_LANG([C])

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
## -- Begin Test for Bluegene Q --

cat > conftest-bgq.cc <<EOF
#include <stdio.h>
int
main()
{
#ifdef __bgq__
  return 0;
#else
  return 1;
#endif
}
EOF

AC_MSG_CHECKING(if this is a BGQ system)
$CXX $CXXFLAGS conftest-bgq.cc -o conftest-bgq

./conftest-bgq

if test $? = 0; then
  AC_MSG_RESULT(yes)
  IS_BGQ=yes
else
  AC_MSG_RESULT(no)
fi

rm -f conftest-bgq.cc conftest-bgq

AC_SUBST(IS_BGQ)

######### -- End Test for Bluegene Q --


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

case $host in
     *-darwin*)
        # New Mac Clang comiler does not support the -faltivec flag...
        altivec="-faltivec"
        if test "$USING_CLANG" = "yes"; then
           altivec="" 
        fi

        # Check to see if we are runnign on OSX version Yosemite or newer:
        OSX_BASE_VERSION=`uname -r | cut -f1 -d"."`
        if [[ $OSX_BASE_VERSION -ge 14 ]]; then  # Note, using double '['s (ie: '[[') because autoconf strips one of them.
           yosemite_plus=yes
        fi

	SO_OR_ARCHIVE_EXTENSION=dylib

        if test "$enable_sci_malloc" = "yes"; then
           AC_MSG_ERROR(sci_malloc does not work under mac osx. Though you can try to use: --enable-sci-malloc=force)
           enable_sci_malloc="no"
        elif test "$enable_sci_malloc" = "force"; then
           enable_sci_malloc="yes"
        fi
	;;
	
     *)
	SO_OR_ARCHIVE_EXTENSION=so 
        ;;
esac

THROW_LEVEL="$enable_throw_level"
extra_cflags=""

if test "$output_avg_walltime" != "no"; then
  extra_cflags="${extra_cflags} -DOUTPUT_AVG_ELAPSED_WALLTIME"
fi

if test "$USING_ICC" = yes; then
  # Turn off the spamming of vectorizing code remark (which comes with icc 2017+)...
  extra_cflags="${extra_cflags} -diag-disable=remark"
fi

LARGESOS=$largesos

if test "$enable_sci_malloc" = "yes"; then
  SCI_MALLOC_ON="yes"
  DEF_MALLOC=""
else
  SCI_MALLOC_ON="no"
  DEF_MALLOC="#define DISABLE_SCI_MALLOC 1"
fi

if test "$with_MallocTrace" != NOT_SET ; then
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
AC_SUBST(LARGESOS)

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

  if test "$USING_ICC" = yes; then
    extra_cflags="-fp-model precise ${extra_cflags}"
    add_important_warning "FYI: Configure added '-fp-model precise' to the optimization flags."
  fi

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

if test ! -e "`which $CC`"; then
  SCI_MSG_ERROR(Could not find the specified C compiler ($CC))
fi
if test ! -e "`which $CXX`"; then
  SCI_MSG_ERROR(Could not find the specified C++ compiler (CC))
fi

if test "$USE_ARCHES" != ""; then
   AC_MSG_ERROR( USE_ARCHES is deprecated.  Please use --enable-arches. )
fi
if test "$USE_FVM" != ""; then
   AC_MSG_ERROR( USE_FVM is deprecated.  Please use --enable-fvm. )
fi
if test "$USE_ICE" != ""; then
   AC_MSG_ERROR( USE_ICE is deprecated.  Please use --enable-ice. )
fi
if test "$USE_MODELS_RADIATION" != ""; then
   AC_MSG_ERROR( USE_MODELS_RADIATION is deprecated.  Models/Radiation will be turned on automatically if Arches or Wasatch are enabled. )
fi
if test "$USE_MPM" != ""; then
   AC_MSG_ERROR( USE_MPM is deprecated.  Please use --enable-mpm. )
fi
if test "$USE_WASATCH" != ""; then
   AC_MSG_ERROR( USE_WASATCH is deprecated.  Please use --enable-wasatch. )
fi

if test "$with_fortran" = "no"; then 

   if test "$BUILD_ARCHES" = "yes"; then
      AC_MSG_ERROR( You may not turn on Arches if you turn off Fortran. )
   fi
   DEF_NO_FORTRAN="#define NO_FORTRAN";
   NO_FORTRAN="yes";
fi

# NO_FORTRAN is used in configVars.mk[.in] and DEF_NO_FORTRAN is used in uintah_testdefs.h.in

AC_SUBST(BUILD_ARCHES)
AC_SUBST(BUILD_FVM)
AC_SUBST(BUILD_ICE)
AC_SUBST(BUILD_MODELS_RADIATION)
AC_SUBST(BUILD_MPM)
AC_SUBST(BUILD_WASATCH)
AC_SUBST(BUILD_HEAT)
AC_SUBST(BUILD_PHASEFIELD)

AC_SUBST(DEF_ARCHES)
AC_SUBST(DEF_FVM)
AC_SUBST(DEF_ICE)
AC_SUBST(DEF_MPM)
AC_SUBST(DEF_NO_FORTRAN)
AC_SUBST(DEF_MODELS_RADIATION)
AC_SUBST(DEF_WASATCH)
AC_SUBST(DEF_HEAT)
AC_SUBST(DEF_PHASEFIELD)
AC_SUBST(NO_FORTRAN)

if test "$BUILD_ARCHES" = "no" -a "$with_fortran" = "no"; then
  sci_required_fortran=no
else
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

##  --  Handle static build stuff... ------------------------

IS_STATIC_BUILD="no"

# If --enable-static, or if lib extension already set to 'a' :
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

if test "$with_MallocTrace" != NOT_SET; then
  AC_MSG_CHECKING(for MallocTrace files (For now we are just verifying the existence of lib and .h files)) 
  if test "$IS_STATIC_BUILD" = "yes"; then
     if test ! -f $with_MallocTrace/lib/libMallocTrace.a; then
     	  AC_MSG_ERROR( '$with_MallocTrace/lib/libMallocTrace.a' does not exist. )
     fi
  else
      if test ! -f $with_MallocTrace/lib/shared/libMallocTrace.so; then
    	   AC_MSG_ERROR( '$with_MallocTrace/lib/shared/libMallocTrace.so' does not exist. )
  	fi
  fi
  if test ! -f $with_MallocTrace/include/MallocTrace.h; then
    AC_MSG_ERROR( '$with_MallocTrace/include/MallocTrace.h' does not exist. )
  fi
  AC_MSG_RESULT(yes)
  DEF_MALLOC_TRACE="#define MALLOC_TRACE 1"
  INC_MALLOC_TRACE_H="-I$with_MallocTrace/include"
  if test "$IS_STATIC_BUILD" = "yes"; then
     MALLOC_TRACE_LIB_DIR_FLAG="$LDRUN_PREFIX$with_MallocTrace/lib -L$with_MallocTrace/lib"
  else								     
     MALLOC_TRACE_LIB_DIR_FLAG="$LDRUN_PREFIX$with_MallocTrace/lib/shared -L$with_MallocTrace/lib/shared"
  fi
  MALLOC_TRACE_LIB_FLAG="-lMallocTrace"
  MALLOC_TRACE_LIB_MPI_FLAG="-lMallocTraceMPI"
fi

AC_SUBST(DEF_MALLOC_TRACE)
AC_SUBST(INC_MALLOC_TRACE_H)
AC_SUBST(MALLOC_TRACE_LIB_FLAG)
AC_SUBST(MALLOC_TRACE_LIB_DIR_FLAG)
AC_SUBST(MALLOC_TRACE_LIB_MPI_FLAG)

#########################################################################
##
## Check for 64 vs 80 bit floating point number storage in register precision error...
## 

case `hostname` in
  titan* )
    AC_MSG_WARN( Skipping 64 vs 80 bit fp number register precision error check on cross compiled machine. )
    ;;
  *)

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

    ;;
esac


#########################################################################
##
##  Add the -fopenmp flag
##

cat > conftest-openmp.cc <<EOF

#include <iostream>

int main()
{
  double a=1;
  double b=.01;
}
EOF

# Verify that the compiler accepts the -fopenmp flag:
$CXX $CXXFLAGS -fopenmp conftest-openmp.cc -o conftest-openmp 2>&5

if test $? = 0; then
   if test "$IS_OSX" = "no"; then
      add_important_warning "Automatically added -fopenmp compiler option to C/CXXFLAGS."
      CFLAGS="$CFLAGS -fopenmp"
      CXXFLAGS="$CXXFLAGS -fopenmp"
   else
      add_important_warning "On Mac: Not adding the openmp compiler option (-fopenmp) to C/CXXFLAGS."
   fi
else
   rm -f conftest-openmp conftest-openmp.cc
   AC_MSG_WARN( Tried to add the -fopenmp flag but the compiler ($CXX) failed to accept it...\n )
fi

rm -f conftest-openmp conftest-openmp.cc

#########################################################################
##
## search for Zoltan
##
if test "$with_zoltan" != NOT_SET; then
  AC_MSG_CHECKING(for Zoltan files (For now we are just verifying the existence of lib and .h files)) 

  if test ! -f $with_zoltan/lib/libzoltan.a; then
    AC_MSG_ERROR( '$with_zoltan/lib/libzoltan.a' does not exist. )
  fi
  if test ! -f $with_zoltan/include/zoltan_cpp.h; then
    AC_MSG_ERROR( '$with_zoltan/include/zoltan_cpp.h' does not exist. )
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
         FFLAGS="-WF, $extra_cflags $FFLAGS"
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
        AC_MSG_WARN( Running under Snow Leopard or later )
        DEF_OSX_SNOW_LEOPARD_OR_LATER="#define OSX_SNOW_LEOPARD_OR_LATER"
        IS_OSX_SNOW_LEOPARD_OR_LATER="yes"
    fi

    NO_LONG_DOUBLE="-Wno-long-double"
    if test "$IS_OSX_SNOW_LEOPARD_OR_LATER" = "yes"; then
      NO_LONG_DOUBLE=
    fi      

    if test $USING_CLANG = "yes"; then
      SCI_CHECK_VERSION($CC,--version,3.0.0,,echo configure: error: not new enough version of $CC.; exit)
      # FIXME: Are these the right flags for Clang on the Mac?"
      CFLAGS="$CFLAGS $extra_cflags -fno-common -dynamic $NO_LONG_DOUBLE"
    else
      if test $USING_GCC = "yes"; then
        SCI_CHECK_VERSION($CC,--version,4.0.0,GCC_VERSION=4.0,)
        if test "$GCC_VERSION" = "4.0"; then
           CFLAGS="$CFLAGS $extra_cflags -fno-common -dynamic $NO_LONG_DOUBLE"
        else
           CFLAGS="$CFLAGS $extra_cflags -fno-common -dynamic $NO_LONG_DOUBLE -fcoalesce-templates"
        fi   
      else
        AC_MSG_WARN( Don't know which type of compiler you are using so cannot verify version... )
      fi
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

  *)
    # linux
    ac_objext=o

    if test "$CC" = "icc"; then
      CFLAGS="$CFLAGS -w1"
      CXXFLAGS="$CXXFLAGS -w1"
    elif test "USING_PGCC" = "yes"; then
      # pgcc doesn't support -Wall (at least not on ranger.tacc.utexas.edu under pgcc v7.1-2)
      CFLAGS=$CFLAGS
      CXXFLAGS=$CXXFLAGS
    elif test "$USING_XLC" = "yes"; then
        # BG/Q xl compilers don't support -Wall (at least not on mira)
      CFLAGS=$CFLAGS
      CXXFLAGS=$CXXFLAGS
    else
      CFLAGS="$CFLAGS -Wall"
      CXXFLAGS="$CXXFLAGS -Wall"
    fi

    ######################################
    #
    # Set -fPIC as necessary...
    #
    if test "$static" != "full"; then
       if test `uname -m` = "ppc"; then
          CFLAGS="-fPIC $CFLAGS"
          FFLAGS="-fPIC $FFLAGS"
          CXXFLAGS="-fPIC $CXXFLAGS"
       # for BG/Q XL compilers
       elif test [`uname -m` = "ppc64" -a $CC != mpicc -a $CC != gcc -a $CC != mpigcc ]; then
          CFLAGS="-qpic $CFLAGS"
          FFLAGS="-qpic $FFLAGS"
          CXXFLAGS="-qpic $CXXFLAGS"
       else
          if test "$enable_64bit" = "yes"; then
             # for BG/Q XL compilers
             if test [`uname -m` = "ppc64" -a $CC != mpicc -a $CC != gcc -a $CC != mpigcc ]; then
               CFLAGS="-qpic $CFLAGS"
               FFLAGS="-qpic $FFLAGS"
               CXXFLAGS="-qpic $CXXFLAGS"
             else
               CFLAGS="-fPIC $CFLAGS"
               FFLAGS="-fPIC $FFLAGS"
               CXXFLAGS="-fPIC $CXXFLAGS"
             fi
          fi
       fi
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

    if test "$USING_GCC" = "yes"; then
      SCI_CHECK_VERSION($CC,--version,2.95.3,,echo configure: error: wrong $CC version.; exit)

      if test "$sci_required_fortran" = "yes"; then
        SCI_CHECK_VERSION($F77,--version,0.5.25,,echo configure: error: wrong $F77 version.; exit)
      fi
      SCI_CHECK_VERSION($CXX,--version,2.95.3,,echo configure: error: wrong $CXX version.; exit)
    else
      if test "$USING_ICC" = "yes"; then
        SCI_CHECK_VERSION($CC,--version,14.0.0,,echo configure: error: wrong $CC version.; exit)
      else
        AC_MSG_WARN(Not checking for version of $CC)
      fi
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
      *xlf*)
         FORTRAN_UNDERSCORE=none
         DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_NONE"
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

# Try -qmakedep (qmakedep option is functionally identical to the -M option.)
# BG/Q specific dpendency generation
for flag in -qmakedep; do
  if test $CC_DEPEND_MODE = unknown; then
    AC_MSG_CHECKING( 1 - if depend mode flag '$flag' works for '$CXX')
    rm -f configure-test.o configure-test-dir/configure-test.o configure-test-dir/configure-test.d
    ac_try="$CXX $CFLAGS -c $flag configure-test-dir/configure-test.d -o configure-test-dir/configure-test.o configure-test-dir/configure-test.cc >/dev/null 2>conftest.out"
    (eval $ac_try) 2>&5
    if test -f configure-test-dir/configure-test.d ; then
      CC_DEPEND_MODE=normal
      CC_DEPEND_REGEN=$flag
      CC_DEPEND_EXT=.d
    else 
      AC_MSG_RESULT( no )
#      echo ""
#      echo "ac_try: $ac_try"   # <= Turn on to debug DEPEND flag issues...
#      echo ""
#      echo "Test results: "
#      echo ""
#      cat conftest.out
#      echo ""
    fi
  fi
done

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
# Uncomment the following block of code to debug DEPEND flag issues...
#      echo ""                 
#      echo "ac_try: $ac_try"   
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

if test $CC_DEPEND_MODE = unknown; then
  AC_MSG_ERROR(Cannot determine dependency generation flag. You might have specified bad '--enable-optimize=' flags on your configure line)
fi

AC_MSG_RESULT(yes.  Mode: $CC_DEPEND_MODE.  Ext: $CC_DEPEND_EXT)
rm -f configure-test.*
rm -rf configure-test-dir

if test "`basename "$F77"`" = "pgf77"; then
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
sci_defs_files="$sci_defs_files include/sci_defs/compile_testdefs.h"

SCI_CHECK_VERSION($MAKE,--version,3.78.1,,
  echo configure: error: wrong GNU make version.; exit)

AC_PROG_AWK

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

if test "$with_fortran" != "no"; then 
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

##  set the default language after the build tools are found
AC_LANG([C])


#########################################################################
##  Set the no_optimize_flag 
##  

case $host in
  *linux*)
    no_optimize_flag=-O0
    ;;
  *)
    ;;
esac


##  ----------------------------------------------------------------------

AC_LANG([C++])
# Each of the HAVE_* statements for these header files is found
# in include/sci_defs/config_testedefs.h
AC_CHECK_HEADERS(limits.h sys/select.h unistd.h sys/time.h ext/algorithm stdint.h inttypes.h)

AC_CONFIG_HEADER(include/sci_defs/config_testdefs.h)
sci_defs_files="$sci_defs_files include/sci_defs/config_testdefs.h"

# Check if we are using a c++-11 compiler

AC_MSG_CHECKING(if using a c++-11 compiler)
have_cxx11="no"

AC_TRY_COMPILE([#include <iostream>],
                [int a; decltype(a) b; auto d = a; auto l = [](){};],
                [have_cxx11=yes],
                [have_cxx11=no])

if test $have_cxx11 = yes; then
   DEF_CXX11="#define HAVE_CXX11 1"
fi   
AC_MSG_RESULT($have_cxx11)
AC_SUBST(DEF_CXX11)

AX_CXX_COMPILE_STDCXX_11

##  ----------------------------------------------------------------------
##  ----------  required checks for required libraries and headers  ------
##  ----------------------------------------------------------------------

##  --  search for the math library  -------------------------------------
AC_LANG(C)

AC_MSG_CHECKING(for math library)

M_LIB_DIR=
M_LIB=m

if test "$with_math" != NOT_SET; then
   M_LIB_DIR=`dirname $with_math`
   # Remove the .a (or .so) and the lib... 
   M_LIB=`basename $with_math | rev | cut -f2- -d"." | rev | cut --complement -c1-3`
fi

AC_MSG_RESULT($M_LIB)

SCI_TRY_LINK(M,math library and header,math.h,,$M_LIB,$M_LIB_DIR,,[
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
   *)
      SCI_MSG_ERROR(unknown thread type for host $host: $threads. exiting.)
      ;;
esac      

AC_SUBST(THREAD_LIB_FLAG)
AC_SUBST(DEF_THREAD)
sci_defs_files="$sci_defs_files include/sci_defs/thread_testdefs.h"

AC_SUBST(SEMAPHORE_LIB_FLAG)

AC_SUBST(CFLAGS)

##  -- check for gcc atomic support

SCI_TRY_LINK(GCCSYNC,gcc style atomic operations ,,,,,,[ int x;   __sync_add_and_fetch(&x,1);],optional)
if test "$HAVE_GCCSYNC" = "yes"; then
  REFCOUNT_IMPL=RefCounted_gcc.cc
else
  REFCOUNT_IMPL=RefCounted.cc
fi
AC_SUBST(REFCOUNT_IMPL)


##  --  Enable Ray Scatter -----------------------------

if test "$ray_scatter" = "yes"; then
  DEF_RAY_SCATTER="#define RAY_SCATTER 1"
fi
AC_SUBST(DEF_RAY_SCATTER)

##  --  Verify CMake -----------------------------

if test "$with_visit" != NOT_SET -o "$build_wasatch_3p" != "no" -o "$build_pidx_lib" != "no"; then
   which cmake > /dev/null 2>&1
   if test $? != 0; then
      AC_MSG_ERROR( Uintah (VisIt/Wasatch3P/PIDX) requires CMake. Please make sure 'cmake' is in your path!)
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
   CMAKE_VERSION_PATCH=`echo "$CMAKE_VERSION_STRING" | awk -F. '{print $3}'`

   if   test "$CMAKE_VERSION_MAJOR" -lt 3; then
      AC_MSG_ERROR(CMake must be version 3.1.0 or better)
   elif test "$CMAKE_VERSION_MINOR" -lt 1; then
      AC_MSG_ERROR(CMake must be version 3.1.0 or better.  $CMAKE_LOCATION is version $CMAKE_VERSION_STRING.)
   #elif test "$CMAKE_VERSION_PATCH" -lt 0; then
   #   AC_MSG_ERROR(CMake must be version 3.1.0 or better.  $CMAKE_LOCATION is version $CMAKE_VERSION_STRING.)
   fi
fi

##  --  search for visit stuff  -------------------------------------------
# local     - src/sim/V2/lib/VisItControlInterface_V2.h
#           - build/lib/libsimV2.a

# installed - linux-x86_64/libsim/V2/include/VisItControlInterface_V2.h
#           - linux-x86_64/libsim/V2/lib/libsimV2.a

_sci_lib=
_sci_inc=
_sci_optional=optional
if test "$sci_required_visit" = "yes"; then
 
  if test "$with_visit" = "yes"; then
    _sci_optional=required
  else
    _sci_optional=specific
    _sci_lib="$with_visit"
    BASE_LIB_PATH(_sci_lib,$_sci_lib)
    # Check for a local version (compiling against source)
    if test -e $with_visit/sim/V2/lib/VisItControlInterface_V2.h; then
      _sci_inc="$with_visit/sim/V2/lib"
    elif test -e $with_visit/../src/sim/V2/lib/VisItControlInterface_V2.h; then
      _sci_inc="$with_visit/../src/sim/V2/lib"
    # Otherwise assume an installed version
    else
      _sci_inc="$with_visit/include"
    fi
  fi

  echo $_sci_inc
  echo $_sci_lib
 
  AC_LANG(C++)
  SCI_TRY_LINK(VISIT, VISIT libraries and headers,
               VisItControlInterface_V2.h VisItDataInterface_V2.h,		
               $_sci_inc,
               simV2,
               $_sci_lib,
               ,,
               $_sci_optional)
  AC_LANG(C)     
  
  if test "$HAVE_VISIT" = "yes"; then
    DEF_VISIT="#define HAVE_VISIT 1"

    # Check for a local version (compiling against source)
    if test -e $with_visit/sim/V2/lib; then
        VISIT_PATH="#define VISIT_PATH ((char*)\"$with_visit\")"
    # Otherwise assume an installed version
    else
        VISIT_PATH="#define VISIT_PATH ((char*)\"${with_visit}/../../../..\")"
    fi
  fi

fi #end sci_required_visit

# these are all defined after the call to SCI_COMPILE_LINK_TEST
AC_SUBST(HAVE_VISIT)
AC_SUBST(DEF_VISIT)
AC_SUBST(INC_VISIT_H)
AC_SUBST(VISIT_LIB_DIR_FLAG)
AC_SUBST(VISIT_LIB_FLAG)
AC_SUBST(VISIT_PATH)

sci_defs_files="$sci_defs_files include/sci_defs/visit_testdefs.h"

##  --  search for OFED (Infiniband Drivers) -----------------------------

if test "$with_ofed" != NOT_SET; then

  dir="$with_ofed"
  BASE_LIB_PATH(DIR,$dir)

  SCI_TRY_LINK( OFED,Infiniband stuff (for MPI),,,
                ibverbs ibumad rdmacm pthread,$DIR,,,required)
  MPI_EXTRA_LIB_FLAG="$MPI_EXTRA_LIB_FLAG $OFED_LIB_FLAG"
  MPI_EXTRA_LIB_DIR="$MPI_EXTRA_LIB_DIR $OFED_LIB_DIR"

fi

##  --  search for MX (myrinet drivers) -----------------------------

if test "$with_mx" != NOT_SET; then

  dir="$with_mx"
  BASE_LIB_PATH(DIR,$dir)

  SCI_TRY_LINK( MX,Myrinet stuff (for MPI),,,myriexpress,$DIR,,,required)
  MPI_EXTRA_LIB_FLAG="$MPI_EXTRA_LIB_FLAG $MX_LIB_FLAG"
  MPI_EXTRA_LIB_DIR="$MPI_EXTRA_LIB_DIR $MX_LIB_DIR"

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
case $host in
  *-darwin*)
    SCI_TRY_LINK(STD_HDRS, standard headers-2,unistd.h dirent.h sys/time.h sys/param.h netdb.h sys/select.h sys/socket.h sys/mman.h sys/ioctl.h sys/resource.h sys/wait.h sys/utsname.h netinet/in.h,,,,,,required)
    ;;
  *)
    # default system headers that cannot be tested on all systems
    SCI_TRY_LINK(STD_HDRS,standard headers-2,unistd.h dirent.h sys/time.h sys/param.h values.h malloc.h netdb.h sys/select.h sys/socket.h sys/mman.h sys/ioctl.h sys/resource.h sys/wait.h sys/utsname.h netinet/in.h,,,,,,required)
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

      xml2_static=`$_sci_lib/../bin/xml2-config | grep -q static`

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
  xmlDocPtr doc = nullptr;       /* document pointer */
  xmlNodePtr root_node = nullptr, node = nullptr, node1 = nullptr;/* node pointers */
  xmlDtdPtr dtd = nullptr;       /* DTD pointer */
  char buff[256];
  int i, j;

  LIBXML_TEST_VERSION;

  /*
    * Creates a new document, a node and set it as a root node
    */
  doc = xmlNewDoc(BAD_CAST "1.0");
  root_node = xmlNewNode(nullptr, BAD_CAST "root");
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


##  --  search for gzopen (and related)  ----------------------------------

_sci_optional=required
_sci_lib=
_sci_inc=

if test "$with_zlib" != NOT_SET; then
  _sci_inc="$with_zlib/include"
  _sci_optional=specific

  # account for /lib64, /usr/lib64, etc
  BASE_LIB_PATH(_sci_lib,$with_zlib)
fi

SCI_TRY_LINK(Z,zlib library and header,zlib.h,$_sci_inc,z,$_sci_lib,,,$_sci_optional)

if test "$HAVE_Z" = "yes"; then
   DEF_Z="#define HAVE_ZLIB 1"
fi

if test "$with_zlib" != NOT_SET && test ! "$DEF_Z"; then
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

if test "$with_jpeg" != NOT_SET; then
  _sci_lib="$with_jpeg/lib"
  _sci_inc="$with_jpeg/include"
  _sci_optional=specific
fi

SCI_TRY_LINK(JPEG,jpeg library,,$_sci_inc,jpeg,$_sci_lib,,,$_sci_optional)

AC_SUBST(JPEG_LIB_FLAG)
AC_SUBST(JPEG_LIB_DIR_FLAG)

_sci_optional=optional
_sci_lib=
_sci_inc=

if test "$with_tiff" != NOT_SET; then
  _sci_lib="$with_tiff/lib"
  _sci_inc="$with_tiff/include"
  _sci_optional=specific
fi

SCI_TRY_LINK(TIFF,tiff library,tiff.h,$_sci_inc,tiff,$_sci_lib,$M_LIB_DIR_FLAG $M_LIB_FLAG,,$_sci_optional)

AC_SUBST(TIFF_LIB_FLAG)
AC_SUBST(TIFF_LIB_DIR_FLAG)
AC_SUBST(HAVE_TIFF)

##  --  check if exceptions work properly --------------------------------

AC_MSG_CHECKING(for working exceptions)


case `hostname` in
  titan* )
    AC_MSG_RESULT(Titan cross compiles so cannot make this check - assuming it works.)
    enable_exceptions_crash=0
    EXCEPTIONS_CRASH_DEF=""
    ;;
  *)
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
    $CXX $CFLAGS -o conftest-exc conftest-exc.cc $LDFLAGS
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

    rm -rf conftest-exc.cc conftest-exc conftest-exc.dSYM
esac
AC_SUBST(EXCEPTIONS_CRASH_DEF)

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

if test "$with_fortran" != NOT_SET -a "$with_fortran" != "no"; then
  _sci_lib="$with_fortran/lib"
  _sci_inc="$with_fortran/include"
  _sci_optional=specific
fi

if test "$with_fortran" != "no"; then
  case $host in 
    *linux*)
      F77TEST=$F77
      case $F77 in
        *mpif77* )
          mpif77 --version | grep -i -q ifort
          if test $? = 0; then
             F77TEST=ifort
          else
             F77TEST=gfortran
          fi
      esac


      # Linux Library Check
      case $F77TEST in
        *pgf77*)
          SCI_TRY_LINK(F,fortran libraries (pgf77),,,pgftnrtl,$_sci_lib,$RT_LIB_FLAG,,$_sci_optional)
          ;;
        *gfortran*)
          SCI_TRY_LINK(F,fortran libraries (gfortran),,,gfortran,$_sci_lib,,,$_sci_optional)
          ;;
        *xlf* )
	  # bgxlf and /usr/local/tools/toolchain-4.7.2/gnu-linux-4.7.2/powerpc64-bgq-linux/lib
          SCI_TRY_LINK(F,fortran libraries (xlf),,,xlf90_r xl xlopt dl,$_sci_lib,,,optional)
          ;;
        *ifort*)
          SCI_TRY_LINK(F,fortran libraries (ifort),,,ifcore,$_sci_lib,,,$_sci_optional)
          ;;
        *ftn*)
          SCI_TRY_LINK(F,fortran libraries (ftn),,,,$_sci_lib,,,$_sci_optional)
          ;;
        *)
          SCI_TRY_LINK(F,fortran libraries (other),,,g2c,$_sci_lib,,,$_sci_optional)
          ;;
       esac
      ;;
    *dec*)
      SCI_TRY_LINK(F,fortran libraries (dec),,,for Ufor,$_sci_lib,,,$_sci_optional)
      ;;
    *-darwin*)
      # OSX Library Check
      if test "$F77" = "xlf"; then
        SCI_TRY_LINK(F,Mac fortran library (xlf) - xlf90,,$_sci_inc,xlf90,$_sci_lib,,,$_sci_optional)
      else
        SCI_TRY_LINK(F,2 - Mac fortran library (gfortran) - gfortran,,$_sci_inc,gfortran,$_sci_lib,,,optional)
        if test "$HAVE_F" = "no"; then
           if test "$_sci_lib" = ""; then
              # At least on Jeremy's new desktop Mac, the gcc compiler does not (apparently) look in /usr/local/lib by default
              # any more.  So if the user has not specified a directory, and the above gfortran test fails, then set sci_lib
              # to /usr/local/lib, and test again.
              _sci_lib="/usr/local/lib"
              SCI_TRY_LINK(F,3 - Mac fortran library (gfortran in /usr/local/lib) - gfortran,,$_sci_inc,gfortran,$_sci_lib,,,optional)
           fi
        fi
        if test "$HAVE_F" = "no"; then
           SCI_TRY_LINK(F,4 - Mac fortran library (g2c) - g2c,,$_sci_inc,g2c,$_sci_lib,,,$_sci_optional)
        fi
      fi
      ;;
    *-pc-mingw*)
      # Mingw g77 and libg2c
      SCI_TRY_LINK(F,fortran libraries (pc-mingw),,,g2c,$_sci_lib,,,$_sci_optional)
      ;;
    *)
      SCI_TRY_LINK(F,fortran libraries (other),,,F77,$_sci_lib,,,$_sci_optional)
      ;;
  esac
fi
  
AC_SUBST(F_LIB_FLAG)
AC_SUBST(F_LIB_DIR_FLAG)

MPI_EXTRA_LIB_FLAG="$MPI_EXTRA_LIB_FLAG $F_LIB_FLAG"
MPI_EXTRA_LIB_DIR="$MPI_EXTRA_LIB_DIR $F_LIB_DIR_FLAG"

AC_LANG(C)

# Clean up all the fortran flags that don't have individual without-fortran guards on them...
#
if test "$with_fortran" = "no"; then 

  FFLAGS="fortran_disabled"
  extra_fflags="fortran_disabled"
  FORTRAN_UNDERSCORE="// FORTRAN_UNDERSCORE not set. Fortran is disabled..."

  DEF_FORTRAN_UNDERSCORE="#define FORTRAN_UNDERSCORE_END /* Fortran disabled, but need for sci_lapack.cc. */"
  msg="Though Fortran is not being used, assuming FORTRAN_UNDERSCORE=end for Core/Math/sci_lapack.cc..."
  add_important_warning "$msg"
fi

##  --  search for acml (optimized blas/lapack) stuff  -------------------------------------

if test "$with_acml" != NOT_SET; then

  if test "$with_blas" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-blas and --with-acml.)
  fi
  if test "$with_atlas" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-atlas and --with-acml.)
  fi
  if test "$with_mkl" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-mkl and --with-acml.)
  fi
  if test "$with_essl" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-essl and --with-acml.)
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

if test "$with_mkl" != NOT_SET; then

  if test "$with_blas" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-blas and --with-mkl.)
  fi
  if test "$with_atlas" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-atlas and --with-mkl.)
  fi
  if test "$with_essl" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-essl and --with-mkl.)
  fi

  _sci_lib="$with_mkl/lib"
  _sci_inc="$with_mkl/include"
  if test ! -e "$with_mkl" -o ! -e "$_sci_lib" -o ! -e "$_sci_inc"; then
     AC_MSG_ERROR(with_mkl (or lib/incldue) dir does not exist: $with_mkl)
  fi
    
  # _sci_optional=specific TESTING FOR SPECIFIC NOT IMPLEMENTED!!!
    
  ## check for lib/intel64 or lib/ia32 on IBM systems (this is the case for MKL, etc)
  if test "$enable_64bit" = "yes"; then
    if test -d $with_mkl/lib/intel64; then
      _sci_lib=$_sci_lib/intel64
    fi
  elif test -d $with_mkl/lib/ia32; then
    _sci_lib=$_sci_lib/ia32
  fi

  case $host in
    *-darwin*)
       AC_MSG_ERROR(We recommend that you don't use MKL on a Mac... use the default vecLib instead.)
       ;;
     *)
       unset test_libs
       test_libs[[0]]="mkl_intel_lp64 mkl_sequential mkl_core pthread" # Line for Utah's Ember computer (10/18/2013)
       test_libs[[1]]="mkl mkl_core mkl_intel_lp64 guide" # Newer version of MKL doesn't have libguide, so need two separate checks...
       test_libs[[2]]="mkl mkl_core mkl_intel_lp64"
       test_libs[[3]]="mkl_core mkl_intel_thread iomp5 pthread" # works on Yellowstone computer
       test_libs[[4]]="mkl_core mkl_intel_lp64 mkl_intel_thread iomp5" # works on Navy DSRC Haise/Kilrain (IBM iDataPlex)

       num_libs=${#test_libs[*]}

       for (( num=0; num<$num_libs; num++ ));
       do
          the_libs=${test_libs[[$num]]}
          SCI_TRY_LINK(BLAS,${num} - mkl version of (blas/lapack) libraries and headers,mkl_cblas.h mkl_lapack.h,
                       $_sci_inc,$the_libs,$_sci_lib,
                       $THREAD_LIB_FLAG $M_LIB_DIR_FLAG $M_LIB_FLAG,[
                                                                    float x[]={0.7f};
                                                                    float y[]={0.8f};
                                                                    float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                                    ],optional)
          if test "$HAVE_BLAS" = "yes"; then
             # FIXME: Really need to do 'specific' test here...
             break
          fi
       done								    

       ;;
  esac

  if test "$HAVE_BLAS" != "yes"; then
    AC_MSG_ERROR(Failed to link against MKL specified by --with_mkl.)
  fi

  HAVE_LAPACK=$HAVE_BLAS

  CBLAS_H="mkl_cblas.h"
  HAVE_CBLAS=$HAVE_BLAS
  DEF_CBLAS="#define HAVE_CBLAS 1"
  DEF_BLAS="#define HAVE_BLAS 1"
  DEF_LAPACK="#define HAVE_LAPACK 1"
  sci_required_blas=no
fi

##  --  search for essl (optimized blas/lapack) stuff  -------------------------------------

if test "$with_essl" != NOT_SET; then

  if test "$with_blas" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-blas and --with-essl.)
  fi
  if test "$with_atlas" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-atlas and --with-essl.)
  fi

  if test "$with_essl" = "yes"; then
    AC_MSG_ERROR(You must specify a directory with --with-essl=DIR)
  elif test "$with_essl"; then
    _sci_lib="$with_essl/lib"
    _sci_inc="$with_essl/include"
    
    # _sci_optional=specific TESTING FOR SPECIFIC NOT IMPLEMENTED!!!
  fi

  case $host in
    *-darwin*)
       AC_MSG_ERROR(We recommend that you don't use ESSL on a Mac... use the default vecLib instead.)
       ;;
     *)
       unset test_libs
       test_libs[[0]]="essl"   # 
       test_libs[[1]]="esslbg" # Line for LLNL's Vulcan computer (3/10/2014)

       num_libs=${#test_libs[*]}

       for (( num=0; num<$num_libs; num++ ));
       do
          the_libs=${test_libs[[$num]]}
          SCI_TRY_LINK(BLAS,${num} - essl version of (blas/lapack) libraries and headers ($the_libs),essl.h,
                       $_sci_inc,dl $the_libs,/usr/local/tools/toolchain-4.7.2/gnu-linux-4.7.2/powerpc64-bgq-linux/lib $_sci_lib,
                       $THREAD_LIB_FLAG $M_LIB_DIR_FLAG $M_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG,
                       [
                       float x[]={0.7f};
                       float y[]={0.8f};
                       float test=sdot(1,x,1,y,-1);
                       ],
                       optional)
          if test "$HAVE_BLAS" = "yes"; then
             # FIXME: Really need to do 'specific' test here...
             break
          fi
       done								    

       ;;
  esac

  HAVE_LAPACK=$HAVE_BLAS

  HAVE_CBLAS=no
  DEF_BLAS="#define HAVE_BLAS 1"
  DEF_LAPACK="#define HAVE_LAPACK 1"
  sci_required_blas=no
fi

##  --  search for atlas (optimized blas/lapack) stuff  -------------------------------------

if test ! "$HAVE_BLAS"; then
  # MKL test above did not find mkl... so look for atlas

  if test "$with_atlas" != NOT_SET -a "$with_blas" != NOT_SET; then
    AC_MSG_ERROR(You cannot use both --with-blas and --with-atlas.)
  fi

  _sci_lib=
  _sci_inc=
  _sci_optional=optional

  if test "$with_atlas" = "yes"; then
    _sci_optional=required
  elif test "$with_atlas" != NOT_SET; then
    _sci_lib="$with_atlas/lib"
    _sci_inc="$with_atlas/include"
    _sci_optional=specific
  fi
  case $host in
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

  if test "$with_blas" != NOT_SET; then
    _sci_lib="$with_blas/lib"
    _sci_inc="$with_blas/include"
    _sci_specific=yes
  fi
  
  DEF_ACCELERATE=""
  HAVE_ACCELERATE=no

  case $host in
    *-darwin*)
       if test "$yosemite_plus" = yes; then
          SCI_TRY_LINK(BLAS,blas and cblas libraries-darwin (yosemite+),Accelerate/Accelerate.h,
                       $_sci_inc,cblas blas,$_sci_lib,-framework Accelerate $F_LIB_DIR_FLAG $F_LIB_FLAG,[
                                                                                float x[]={0.7f};
                                                                                float y[]={0.8f};
                                                                                float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                                                ],optional)
          if test "$HAVE_BLAS" = yes ; then
             BLAS_LIB_FLAG="-framework Accelerate"
             CBLAS_H="Accelerate/Accelerate.h"
             HAVE_CBLAS=yes
             DEF_CBLAS="#define HAVE_CBLAS 1"
             HAVE_ACCELERATE=yes
             DEF_ACCELERATE="#define HAVE_ACCELERATE 1"
          fi
       else
          SCI_TRY_LINK(BLAS,blas and cblas libraries-darwin,vecLib/cblas.h,
                       $_sci_inc,cblas blas,$_sci_lib,-framework vecLib $F_LIB_DIR_FLAG $F_LIB_FLAG,[
                                                                                float x[]={0.7f};
                                                                                float y[]={0.8f};
                                                                                float test=cblas_sdsdot(1,0.0,x,1,y,-1); 
                                                                                ],optional)
          if test "$HAVE_BLAS" = yes ; then
             BLAS_LIB_FLAG="-framework vecLib"
             CBLAS_H="vecLib/cblas.h"
             HAVE_CBLAS=yes
             DEF_CBLAS="#define HAVE_CBLAS 1"
	     HAVE_ACCELERATE=yes
             DEF_ACCELERATE="#define HAVE_ACCELERATE 1"
          fi
       fi
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

  if test "$with_blas" != NOT_SET && test ! "$DEF_BLAS"; then
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

AC_SUBST(HAVE_ACCELERATE)
AC_SUBST(DEF_ACCELERATE)

sci_defs_files="$sci_defs_files include/sci_defs/blas_testdefs.h"

##  --  search for lapack stuff  -----------------------------------------

if test "$HAVE_LAPACK" != "yes"; then
  # if HAVE_LAPACK is yes at this point, then atlas/mkl has been found and we
  # don't want to check for it here.

  if test "$with_lapack" != NOT_SET; then
     _sci_lib="$with_lapack/lib"
  else
     _sci_lib="$LIB_DIR_BLAS"
  fi

  case $host in
    *-darwin*)

       if test "$yosemite_plus" = yes; then
          SCI_TRY_LINK(LAPACK,Mac Accelerate lapack version of libraries and headers,
                       Accelerate/Accelerate.h,,,,-framework Accelerate,,optional)
          if test "$HAVE_LAPACK" = yes ; then
             LAPACK_LIB_FLAG="-framework Accelerate"
          fi
       else
          SCI_TRY_LINK(LAPACK,Old Mac lapack version of libraries and headers,
                       vecLib/clapack.h,,,,$altivec -framework vecLib,,optional)
          if test "$HAVE_LAPACK" = yes ; then
             LAPACK_LIB_FLAG="$altivec -framework vecLib"
          fi
       fi
       SCI_TRY_LINK(LAPACK,lapack library,,,lapack,$_sci_lib,$BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG,,optional)
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

  if test "$HAVE_LAPACK" = "yes"; then
    DEF_LAPACK="#define HAVE_LAPACK 1"
  fi

  if test "$sci_required_lapack" = "yes" -o "$with_lapack" != NOT_SET && test ! "$DEF_LAPACK"; then
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

if test "$with_mpi_extra_libs" != NOT_SET; then
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

if test $with_mpi = NOT_SET -a $with_mpi_lib = NOT_SET -a $with_mpi_include = NOT_SET ; then
  AC_MSG_ERROR( [MPI is requried for Uintah.  You must use either --with-mpi or --with-mpi-lib/include.  If you are using an 'mpicc' compiler, then you should configure with: "--with-mpi=built-in".] )
fi

if test "$with_mpi" = "built-in"; then

  if test "$with_mpi_include" != NOT_SET -o "$with_mpi_lib" != NOT_SET; then
    AC_MSG_ERROR( [Please use either --with-mpi or --with-mpi-lib/include, but not both.] )
  fi

  HAVE_MPI=yes
  AC_MSG_CHECKING(for location of mpi.h (as MPI was specified as built-in))

  ########################################################################
  ## Determine mpi.h location (it is needed by nvcc)
  cat > conftest-uintah-mpi.cc <<EOF
#include <mpi.h>
int main(int argc, char **argv)
{
  return 0;
}
EOF
  $CC -c $CC_DEPEND_REGEN  conftest-uintah-mpi.cc > /dev/null 2>&1
  if test -f conftest-uintah-mpi.d; then
    mpi_h=`grep "/mpi.h" conftest-uintah-mpi.d | head -1 | sed 's,^ *,,g' | cut -f1 -d" "`
    mpi_h_dir=`dirname $mpi_h`
    if test -d $mpi_h_dir; then
      INC_MPI_H_NVCC=-I$mpi_h_dir
      AC_MSG_RESULT($INC_MPI_H_NVCC)
    else
      AC_MSG_RESULT(WARNING: Found bad mpi.h location of: $mpi_h_dir -- nvcc may have issues!)
    fi
  else
    AC_MSG_RESULT(WARNING: Could not determine location of mpi.h -- nvcc may have issues!)
  fi
  rm -rf conftest-uintah-mpi*
fi

########################################################################
# MPI not built-in, so look for it...

if test "$sci_required_mpi" = "yes" -a "$with_mpi" != "built-in"; then
  echo
  AS_MESSAGE(checking for required MPI components......)
  echo

  if test "$with_mpi" != NOT_SET -a \( "$with_mpi_include" != NOT_SET -o "$with_mpi_lib" != NOT_SET \); then
    AC_MSG_ERROR( [Please use either --with-mpi or --with-mpi-lib/include, but not both.] )
  fi
  if test \( "$with_mpi_lib" != NOT_SET -a "$with_mpi_include" = NOT_SET \) -o \( "$with_mpi_lib" = NOT_SET -a "$with_mpi_include" != NOT_SET \); then
    AC_MSG_ERROR( [Both --with-mpi-lib=DIR and --with-mpi-include=DIR must be specified...] )
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
        if test "$with_mpi" != NOT_SET; then
           _sci_optional=specific
           _sci_inc="$with_mpi/include"
           # Account for lib64 or lib32 or just lib...
           BASE_LIB_PATH(_sci_lib,$with_mpi)
        else
          if test "$with_mpi_lib" != NOT_SET; then
            _sci_inc="$with_mpi_include"
            _sci_lib="$with_mpi_lib"
          fi
        fi

        unset test_libs
        test_libs[[0]]="mpi lam util pmpi" 
        test_libs[[1]]="mpi lam util $THREAD_LIB_FLAG"
        test_libs[[2]]="lam mpi"                         # Newer LAM on OSX, pthreads and util not needed.
        test_libs[[3]]="mpi"
        test_libs[[4]]="mpich_threadm mpl"               # Try mpich with mpich_threadm and mpl library.
        test_libs[[5]]="mpich_threadm"                   # Same as above, but wihtout mpl
        test_libs[[6]]="mpich mpl pmpich"
        test_libs[[7]]="mpich mpl"
        test_libs[[8]]="mpich pmpich"
        test_libs[[9]]="mpich"
        test_libs[[10]]="mpi_mt"
	test_libs[[11]]="mpich-xl opa-xl mpl-xl pami-gcc SPI SPI_cnk rt pthread stdc++" # Vulcan@LLNL

        unset other
        other[[0]]=""
        other[[1]]="-DMPIPP_H"
        other[[2]]="-DMPIPP_H"
        other[[3]]="-DMPIPP_H"
        other[[4]]=""
        other[[5]]=""
        other[[6]]=""
        other[[7]]=""
        other[[8]]=""
        other[[9]]=""
        other[[10]]=""
        other[[11]]=""

        num_libs=${#test_libs[*]}

        for (( num=0; num<$num_libs; num++ )); do

           # After each attempt to locate mpi, we need to verify that the mpi it
           # chose is in the path we specified.  This is because some machines
           # have MPI installed in a standard location and sometimes will choose
           # that one over the one we specified.

           the_libs=${test_libs[[$num]]}
           other=${other[[$num]]}

           the_file="lib`echo "${test_libs[[$num]]}" | cut -f1 -d" "`"

           SCI_TRY_LINK(MPI,${num}: mpi library and header ($the_libs),mpi.h,
                        $_sci_inc, $the_libs $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,$other,,optional)

           if test "$HAVE_MPI" = "yes"; then
              if test "$_sci_optional" = "specific" &&
                 test ! -e $_sci_lib/$the_file.so &&
                 test ! -e $_sci_lib/$the_file.a &&
                 test ! -e $_sci_lib/$the_file.dylib; then
                    AC_MSG_WARN(Found libmpi but not in requested directory: $_sci_lib)
                    HAVE_MPI="no"
              else
                 break
              fi
           fi
        done

	######################################################################
	# If we didn't find mpi above, try it again, but this time hack in 
	# /usr/include/mpi/ into the path.  (This is necessary as some (many?)
	# linux distributions place the mpi lib in /usr/lib, but put mpi.h
	# in /usr/include/mpi/... The problem with hardcoding /usr/include/mpi
	# into the link test path is that it can (in some situations) cause
	# the wrong mpi.h to be used with non-matching libraries...
	# Not sure I like this hack, but don't have a better fix off hand. -Dd

        if test "$HAVE_MPI" != "yes"; then
          _sci_inc="$_sci_inc /usr/include/mpi"
          for (( num=0; num<$num_libs; num++ )); do

            # After each attempt to locate mpi, we need to verify that the mpi it
            # chose is in the path we specified.  This is because some machines
            # have MPI installed in a standard location and sometimes will choose
            # that one over the one we specified.

            the_libs=${test_libs[[$num]]}
            other=${other[[$num]]}

            the_file="lib`echo "${test_libs[[$num]]}" | cut -f1 -d" "`"

            SCI_TRY_LINK(MPI,${num}: mpi library and header (mpi.h hack) ($the_libs),mpi.h,
                         $_sci_inc, $the_libs $MPI_EXTRA_LIB_FLAG,$_sci_lib $MPI_EXTRA_LIB_DIR,$other,,optional)

            if test "$HAVE_MPI" = "yes"; then
              if test "$_sci_optional" = "specific" &&
                 test ! -e $_sci_lib/$the_file.so &&
                 test ! -e $_sci_lib/$the_file.a &&
                 test ! -e $_sci_lib/$the_file.dylib; then
                    AC_MSG_WARN(Found libmpi but not in requested directory: $_sci_lib)
                    HAVE_MPI="no"
              else
                 break
              fi
            fi
          done
        fi
	######################################################################
        
  esac
  
  MPI_LIB_FLAG="$MPI_LIB_FLAG $MALLOC_TRACE_LIB_MPI_FLAG"
  MPI_LIB_DIR_FLAG="$MPI_LIB_DIR_FLAG $MALLOC_TRACE_LIB_DIR_FLAG"
fi #end sci_required_mpi

########################################################################
## Verify that MPI uses const correctly.. Mira check.
AC_MSG_CHECKING(if MPI uses const correctly)
cat > mpi_const_test.cc << EOF
#include <mpi.h>
inline int Add_error_string( int errorcode , const char *string )
{
  return MPI_Add_error_string( errorcode , string );
}
EOF

$CC $INC_MPI_H -c mpi_const_test.cc > /dev/null 2>&1
if test $? != 0; then
   DEF_MPI_CONST_WORKS="#define MPICONST "
   AC_MSG_RESULT(no)
else
   DEF_MPI_CONST_WORKS="#define MPICONST const"
   AC_MSG_RESULT(yes)
fi

# Clean up
rm -rf mpi_const_test*

# FIXME: These two defines need to be tested for and set correctly:
DEF_MPI_MAX_THREADS="#define MPI_MAX_THREADS 64"
DEF_MPI3_ENABLED="#define UINTAH_ENABLE_MPI3 false"

rm -f mpi_const_test.cc mpi_const_test.o

AC_SUBST(HAVE_MPI)
AC_SUBST(DEF_MPI_CONST_WORKS)
AC_SUBST(DEF_MPI_MAX_THREADS)
AC_SUBST(DEF_MPI3_ENABLED)
AC_SUBST(INC_MPI_H)
AC_SUBST(INC_MPI_H_NVCC)
AC_SUBST(MPI_LIB_FLAG)
AC_SUBST(MPI_LIB_DIR_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/mpi_testdefs.h"

#  --  search for PETSc  --------------------------------------------
AC_LANG([C++])
if test "$with_petsc" = "no"; then # Handle --without option
  AC_MSG_WARN( "PETSc disabled." )
else  

  if test "$with_petsc" = "built-in"; then
     HAVE_PETSC="yes"
     DEF_PETSC="#define HAVE_PETSC 1"
  else
    # Make sure we have either mpi, blas, and lapack
    # (It seems that under OSF, the libs that cover blas also cover lapack)
    if test "$HAVE_MPI" = "yes" -a "$HAVE_BLAS" = "yes" -a "$HAVE_LAPACK" = "yes"; then
      _sci_lib_dir=

      if test "$with_petsc" != NOT_SET; then
         version_file=$with_petsc/include/petscversion.h
         _sci_inc="$with_petsc/include"
      else
         version_file=/usr/include/petsc/petscversion.h
         lib_dir=/usr
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

        if [[[ $CC = *mpigcc* && $IS_BGQ = yes ]]]; then
           msg="Updating CFLAGS with PETSc required flags due to using xlC compiled PETSC with gcc compiler..."
           add_important_warning "$msg"
           #CFLAGS="  $CFLAGS   -DPETSC_RESTRICT=__restrict__ -DPETSC_DEPRECATED(why)= "
	   ORIG_CXXFLAGS=$CXXFLAGS
           CXXFLAGS="$CXXFLAGS -DPETSC_RESTRICT=__restrict__ -DPETSC_DEPRECATED(why)= "
        fi

        SCI_TRY_LINK( PETSC,
                      petsc libraries and headers - multi-lib version,
                      petsc.h,
                      $_sci_inc,
                      petsc petscvec petscmat petscdm petscsnes petscts petscksp,
                      $_sci_lib_dir,
                      $INC_MPI_H -DMPIPP_H $LAPACK_LIB_DIR_FLAG $LAPACK_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $MALLOC_TRACE_LIB_DIR_FLAG $MALLOC_TRACE_LIB_FLAG $M_LIB_DIR_FLAG $M_LIB_FLAG $MPI_LIB_DIR_FLAG $MPI_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG $DL_LIB_FLAG $THREAD_LIB_FLAG,
                      ,
                      optional)

        # Test a number of petsc library options if the above didn't work
        if test "$HAVE_PETSC" != "yes"; then
           unset test_libs
           test_libs[[0]]="petsc superlu_dist_3.3 amd.2.2.2 dmumps superlu_4.3 umfpack triangle spai parmetis chaco metis mumps_common scalapack pord"         # Vulcan (tempoary?)
           test_libs[[1]]="petsc"                       # standard single petsc lib
           test_libs[[2]]="craypetsc_intel_real"        # cray petsc-real - for DOD Open Research Systems 
           test_libs[[3]]="craypetsc_intel_complex"     # cray petsc-complex - for DOD Open Research Systems 
                                                                  
           num_libs=${#test_libs[*]}

           for (( num=1; num<=$num_libs; num++ ));
           do
               the_libs=${test_libs[[$num-1]]}
               SCI_TRY_LINK(PETSC,
                            petsc library and header - try $num ($the_libs),
                            petsc.h,
                            $_sci_inc,
                            $the_libs,
                            $_sci_lib_dir,
                            $INC_MPI_H -DMPIPP_H $LAPACK_LIB_DIR_FLAG $LAPACK_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $MALLOC_TRACE_LIB_DIR_FLAG $MALLOC_TRACE_LIB_FLAG $M_LIB_DIR_FLAG $M_LIB_FLAG $MPI_LIB_DIR_FLAG $MPI_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG $DL_LIB_FLAG $THREAD_LIB_FLAG,
                            ,
                            optional)
               # bail if a suitable library is found
               if test "$HAVE_PETSC" = "yes"; then
                  break
               fi
           done 
        fi

        # Reset CFLAGS as necessary... 
        #
        # This hack is necessary because we are using the xlc compiled Petsc on
        # Vulcan with the gcc compilers... And the compilicated CFLAGS will not
        # pass through configure into configVars.mk correctly... sigh.
        #
        # Note, the above "[[[" turns into the desired "[[" once passed through autoconf...
        #
        if [[[ $CC = *mpigcc* && $IS_BGQ = yes ]]]; then
           CXXFLAGS=$ORIG_CXXFLAGS
        fi

        # report error message
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

AC_LANG([C])


##  --  search for xalan  ---------------------------------------------------
if test "$sci_required_xalan" ; then
  AC_MSG_CHECKING(for xalan-j.)

  if test "$with_xalan" != NOT_SET; then
    dir="$with_xalan"
  fi
  if test -f $dir/bin/xalan.jar -a -f $dir/bin/xml-apis.jar -a -f $dir/bin/xercesImpl.jar; then
	AC_MSG_RESULT($dir)
	XALAN_PATH="$dir/bin/xalan.jar:$dir/bin/xml-apis.jar:$dir/bin/xercesImpl.jar"
        DEF_XALAN="#define XALAN 1"
  else	
	AC_MSG_RESULT(not found)
  fi

  if test "$with_xalana" != NOT_SET && test ! "$DEF_XALAN"; then
    echo
    AC_MSG_ERROR(One or more of the XALAN components is missing.)
    echo
  fi

fi

AC_SUBST(XALAN_PATH)

##  --  search for papi stuff  -------------------------------------------

if test "$with_papi" != NOT_SET; then
 
  _sci_optional=specific
  _sci_lib="$with_papi"
  BASE_LIB_PATH(_sci_lib,$_sci_lib)
  _sci_inc="$with_papi/include"
 
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

fi

# these are all defined after the call to SCI_COMPILE_LINK_TEST
AC_SUBST(HAVE_PAPI)
AC_SUBST(DEF_PAPI)
AC_SUBST(INC_PAPI_H)
AC_SUBST(PAPI_LIB_DIR_FLAG)
AC_SUBST(PAPI_LIB_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/papi_testdefs.h"

##  --  search for google perftools stuff  -------------------------------------------

if test "$with_gperftools" != NOT_SET; then

  _sci_optional=specific
  _sci_lib="$with_gperftools"
  # account for lib64 or lib32 or just lib...
  BASE_LIB_PATH(_sci_lib,$_sci_lib)
  _sci_inc="$with_gperftools/include"
  
  if test "$IS_OSX" = "yes"; then
    gperf_lib_list=tcmalloc_and_profiler
  elif test "$IS_STATIC_BUILD" = "yes"; then 
    gperf_lib_list="tcmalloc profiler unwind"
  else  
    gperf_lib_list="tcmalloc profiler"
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

fi

AC_SUBST(HAVE_GPERFTOOLS)
AC_SUBST(DEF_GPERFTOOLS)
AC_SUBST(INC_GPERFTOOLS_H)
AC_SUBST(GPERFTOOLS_LIB_DIR_FLAG)
AC_SUBST(GPERFTOOLS_LIB_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/gperftools_testdefs.h"

##  --  search for ipm stuff  -------------------------------------------
_sci_lib=
_sci_inc=
_sci_optional=optional
if test "$with_ipm" != NOT_SET; then

  if test "$static" != "yes"; then
    AC_MSG_ERROR( [Only use \"--with-ipm\" for static builds. For dynamic usage use: \"LD_PRELOAD=/path/to/ipm/lib/libipm.so\" ] )
  fi
 
  _sci_optional=specific
  _sci_lib="$with_ipm"
  BASE_LIB_PATH(_sci_lib,$_sci_lib)
  _sci_inc="$with_ipm/include"
 
  AC_LANG(C++)
  SCI_TRY_LINK(IPM,
               IPM library,
               ,
               ,
               ipm,
               $_sci_lib,
               ,
               [],
               $_sci_optional)
  AC_LANG(C)     
  
  DEF_IPM="#define USE_IPM 1"

fi

# these are all defined after the call to SCI_COMPILE_LINK_TEST
AC_SUBST(HAVE_IPM)
AC_SUBST(DEF_IPM)
AC_SUBST(INC_IPM_H)
AC_SUBST(IPM_LIB_DIR_FLAG)
AC_SUBST(IPM_LIB_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/ipm_testdefs.h"


##  --  search for cuda stuff  ------------------------------------------
if test "$with_cuda" != NOT_SET; then

  AC_MSG_CHECKING(for required CUDA components)
  echo ""

  # Initialize Uintah's major CUDA make definitions:
  DEF_CUDA=""
  HAVE_CUDA=""
  HAVE_CUDA_H=""
  INC_CUDA_H=""
  CUDA_LIB_DIR=""

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
# test CUDA libraries and headers
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

  # for a static build, use the static CUDA runtime lib
  if test "$static" != "no" -o "$SO_OR_ARCHIVE_EXTENSION" = "a"; then
     _cudart_lib="cudart_static dl pthread rt"
  else
     _cudart_lib="cudart"
  fi

  SCI_CUDA_COMPILE_LINK_TEST(CUDA,
                             CUDA libraries and headers,
                             cuda.h,
                             $_cuda_inc,
                             $_cudart_lib,
                             $_cuda_lib,
                             ,
                             cuda_conf_test.cu,
                             optional,
                             yes)

  rm -rf cuda_conf_test*

fi

# these are all defined after the call to SCI_CUDA_COMPILE_LINK_TEST
AC_SUBST(HAVE_CUDA)
AC_SUBST(DEF_CUDA)
AC_SUBST(INC_CUDA_H)
AC_SUBST(CUDA_LIB_DIR_FLAG)
AC_SUBST(CUDA_LIB_FLAG)
AC_SUBST(NVCC_CFLAGS)
AC_SUBST(NVCC_CXXFLAGS)
AC_SUBST(NVCC_LIBS)

sci_defs_files="$sci_defs_files include/sci_defs/cuda_testdefs.h"


##  --  search for magma stuff  -------------------------------------------

if test "$sci_required_magma" = "yes"; then
  
  # guard against building MAGMA without CUDA, need cublas, etc from CUDA installation
  if test "$with_cuda" = NOT_SET; then
    AC_MSG_ERROR( [Cannot build with MAGMA support without CUDA. Both --with-magma=DIR and --with-cuda=DIR must be specified.] )
  fi

  _sci_optional=required
  _magma_lib=
  _magma_inc=
  _cuda_lib=
  _cuda_inc=

  # account for lib64 or lib32 or just lib...
  BASE_LIB_PATH(_magma_lib,$with_magma)
  BASE_LIB_PATH(_cuda_lib,$with_cuda)   
  _magma_inc="$with_magma/include"
  _cuda_inc="$with_cuda/include"
 
  AC_LANG(C++)
  SCI_TRY_LINK(MAGMA, MAGMA libraries and headers,
               magma.h,
               $_magma_inc $_cuda_inc,
               magma magmablas lapack cublas cudart,
               $_magma_lib $_cuda_lib,
               ,
               [magma_init();
                magma_finalize();],
               $_sci_optional)
  AC_LANG(C)     
  
  DEF_MAGMA="#define HAVE_MAGMA 1"

#[aph] - debug
#echo
#echo "HAVE_MAGMA         = " $HAVE_MAGMA
#echo "HAVE_MAGMA_H       = " $HAVE_MAGMA_H
#echo "INC_MAGMA_H        = " $INC_MAGMA_H
#echo "MAGMA_LIB_FLAG     = " $MAGMA_LIB_FLAG
#echo "MAGMA_LIB_DIR      = " $MAGMA_LIB_DIR
#echo "MAGMA_LIB_DIR_FLAG = " $MAGMA_LIB_DIR_FLAG
#echo

fi #end sci_required_magma

# these are all defined after the call to SCI_COMPILE_LINK_TEST
AC_SUBST(HAVE_MAGMA)
AC_SUBST(DEF_MAGMA)
AC_SUBST(INC_MAGMA_H)
AC_SUBST(MAGMA_LIB_DIR_FLAG)
AC_SUBST(MAGMA_LIB_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/magma_testdefs.h"

##  --  search for visus  -------------------------------------------------
AC_LANG(C++)

if test "$with_visus" != NOT_SET; then
  _sci_lib="$with_visus"
  _sci_inc="$with_visus/include $with_visus"
  BASE_LIB_PATH(_sci_lib,$_sci_lib)
  _sci_optional=required

  if test "$IS_OSX" = "yes"; then
     VISUS_OSX="-framework Foundation"
  fi

  SCI_TRY_LINK(VISUS,visus library and headers,visuscpp/kernel/visus_kernel.h,$_sci_inc,ttmath visusidx curl libz FreeImage tinyxml visusdb visusscenegraph visuskernel curl visusdataflow $VISUS_OSX,$_sci_lib,,,$_sci_optional)

  if test "$HAVE_VISUS" = "yes"; then
     DEF_VISUS="#define HAVE_VISUS 1"
  fi
fi

AC_SUBST(HAVE_VISUS)
AC_SUBST(DEF_VISUS)
AC_SUBST(INC_VISUS_H)
AC_SUBST(VISUS_LIB_FLAG)
AC_SUBST(VISUS_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/visus_testdefs.h"
AC_LANG(C)

##  --  Handle --enable-pidx... ------------------------

if test "$build_pidx_lib" != "no"; then

  if test "$with_pidx" != NOT_SET; then
     AC_MSG_ERROR( Do NOT specify both --with-pidx and --enable-pidx )
  fi

  PIDX_DIR="PIDX_Library"

  $path_to_src/build_scripts/build_pidx.sh `$sci_pwd`/$PIDX_DIR "$debug" "$IS_STATIC_BUILD"

  if test $? != 0; then
     AC_MSG_ERROR( Failed to build PIDX library... )
  fi

  _sci_lib=`$sci_pwd`/$PIDX_DIR/install/lib
  _sci_inc=`$sci_pwd`/$PIDX_DIR/install/include
  _sci_optional=required

  SCI_TRY_LINK(PIDX,pidx library and headers,PIDX.h,$_sci_inc $INC_MPI_H,$MPI_LIB_FLAG pidx,$MPI_LIB_DIR_FLAG $_sci_lib,-DMPIPP_H,,$_sci_optional)

fi

##  --  Verify PIDX lib dir is valid  -------------------------------------------------
AC_LANG(C++)

# Default values:
_sci_optional=optional
_sci_lib=
_sci_inc=

if test "$with_pidx" != NOT_SET; then
  _sci_lib="$with_pidx"
  _sci_inc="$with_pidx/include"
  BASE_LIB_PATH(_sci_lib,$_sci_lib)
  _sci_optional=required

  SCI_TRY_LINK(PIDX,pidx library and headers,PIDX.h,$_sci_inc $INC_MPI_H,$MPI_LIB_FLAG pidx,$MPI_LIB_DIR_FLAG $_sci_lib,-DMPIPP_H,,$_sci_optional)

fi

if test "$HAVE_PIDX" = "yes"; then
   DEF_PIDX="#define HAVE_PIDX 1"
fi

AC_SUBST(HAVE_PIDX)
AC_SUBST(DEF_PIDX)
AC_SUBST(INC_PIDX_H)
AC_SUBST(PIDX_LIB_FLAG)
AC_SUBST(PIDX_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/pidx_testdefs.h"
AC_LANG(C)

##  --  search for Boost ------------------------

AC_LANG([C++])

BOOST_INCLUDE_DIR=
BOOST_LIBRARY_DIR=

if test "$with_boost" != NOT_SET; then

   # Remove the trailing / from with_boost (if any)
   with_boost=${with_boost%/}

   if test "$with_boost_library" != NOT_SET -o "$with_boost_include" != NOT_SET ; then
      AC_MSG_ERROR( You cannot use --with-boost and --with-boost-include/--with-boost-library. )
   elif test "$with_boost" != "/usr"; then
      # Do not put /usr in the path, it is automatically there.

      if test "$USING_GCC" = "yes" -o "$USING_ICC" = "yes"; then
         # If configure specifies a boost dir, then ignore the system dirs to avoid 
         # finding the wrong version of boost.  I know that "--sysroot" works with gcc and icc...

	if test "$IS_OSX" = "no"; then
         ignore_sys_libs="--sysroot=/junk"
	fi
      fi

      BOOST_INCLUDE_DIR=$with_boost/include
      BOOST_LIBRARY_DIR=$with_boost/lib
   fi
fi

if test "$with_boost_library" != NOT_SET -o "$with_boost_include" != NOT_SET; then
   if test "$with_boost_library" = NOT_SET -o "$with_boost_include" = NOT_SET; then
      AC_MSG_ERROR( You must specify both --with-boost-include and --with-boost-library. )
   fi
   # Following magic line removes any trailing / from with_boost_include.
   inc=${with_boost_include%/}
   if test "$inc" = "/usr/include" ; then
      with_boost_include=
   fi
   BOOST_INCLUDE_DIR=$with_boost_include
   BOOST_LIBRARY_DIR=$with_boost_library
fi

if test "$with_boost" != NOT_SET -o "$with_boost_library" != NOT_SET; then

  _boost_incl="boost/signal.hpp boost/shared_ptr.hpp boost/bind.hpp boost/signal.hpp boost/ref.hpp boost/graph/adjacency_list.hpp boost/graph/graph_traits.hpp boost/graph/topological_sort.hpp boost/graph/graphviz.hpp boost/graph/breadth_first_search.hpp boost/graph/visitors.hpp boost/graph/transpose_graph.hpp boost/any.hpp boost/filesystem.hpp"

  _boost_libs_mt="boost_thread-mt boost_signals-mt boost_serialization-mt boost_system-mt boost_program_options-mt boost_filesystem-mt boost_date_time-mt"

  _boost_libs="boost_thread boost_signals boost_serialization boost_system boost_program_options boost_filesystem boost_date_time"

  SCI_TRY_LINK(BOOST,[boost library (mt version) and include headers],$_boost_incl,$BOOST_INCLUDE_DIR,$_boost_libs_mt,$BOOST_LIBRARY_DIR,$ignore_sys_libs,,optional)

  #if test "$HAVE_BOOST" = "yes" -a "$with_boost" != "/usr"; then
  #  if test ! -e "$BOOST_LIBRARY_DIR"/libboost_thread-mt.so -a ! -e "$BOOST_LIBRARY_DIR"/libboost_thread-mt.a; then
  #     AC_MSG_WARN( [Boost was found, but not in the right place... trying again.] )
  #     HAVE_BOOST=no
  #  fi
  #fi

  if test "$HAVE_BOOST" = "no"; then
     SCI_TRY_LINK(BOOST, boost library (non-mt) and include headers,$_boost_incl,$BOOST_INCLUDE_DIR,$_boost_libs,$BOOST_LIBRARY_DIR,,,optional)

     if test "$HAVE_BOOST" = "yes" -a "$with_boost" != "/usr"; then
        if test ! -e "$BOOST_LIBRARY_DIR"/libboost_thread.so -a ! -e "$BOOST_LIBRARY_DIR"/libboost_thread.a; then
           AC_MSG_ERROR( Boost was found but not in the right place... )
        fi
     fi
  fi

  if test "$HAVE_BOOST" = "yes"; then
     DEF_BOOST="#define HAVE_BOOST 1"              
  else
     AC_MSG_ERROR( Did not find boost in: '$with_boost'. )
  fi

  if test "$with_boost" != NOT_SET -a "$with_boost" != "/usr"; then
     # If with_boost points to /usr, then we don't need to explicitly point to it...
     INC_BOOST_H="-I$with_boost/include"
     if test "$IS_STATIC_BUILD" != "yes"; then
        BOOST_LIB_DIR_FLAG="$LDRUN_PREFIX$BOOST_LIB_DIR -L$BOOST_LIB_DIR"
     fi
  fi

  SCI_TRY_LINK( BOOST_FS_V3, boost filesystem v3 namespace...,
                $_boost_incl,$BOOST_INCLUDE_DIR,,,
                $BOOST_LIB_DIR_FLAG $BOOST_LIB_FLAG,
                [  boost::system::error_code ec;
                   boost::filesystem3::copy("abc","def", ec);  ],
                optional)
  if test "$HAVE_BOOST_FS_V3" = "yes"; then
     DEF_BOOST_FILESYSTEM_NAMESPACE_V3="#define UINTAH_BOOST_FILESYSTEM_NAMESPACE_V3 1"
  fi

fi
AC_SUBST(HAVE_BOOST)
AC_SUBST(DEF_BOOST)
AC_SUBST(DEF_BOOST_FILESYSTEM_NAMESPACE_V3)
AC_SUBST(INC_BOOST_H)
AC_SUBST(BOOST_LIB_FLAG)
AC_SUBST(BOOST_LIB_DIR_FLAG)
AC_LANG([C])

sci_defs_files="$sci_defs_files include/sci_defs/boost_testdefs.h"

##  --  search for Hash Map ------------------------

AC_LANG([C++])
AC_MSG_CHECKING(for location of hash_map)
DEF_HASHMAP=None

if test "$DEF_HASHMAP" = "None"; then
  AC_TRY_COMPILE([#include <unordered_map>
                 std::unordered_map<int, int> xx;], ,
		 [DEF_HASHMAP=HAVE_C11_HASHMAP])
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

##  --  Handle --enable-wasatch_3p... ------------------------

if test "$build_wasatch_3p" != "no"; then

  if test "$HAVE_BOOST" != "yes"; then
      AC_MSG_ERROR( Option --enable-wasatch_3p requires Boost... <--with-boost=DIR> )
  fi
  if test "$with_wasatch3p" != NOT_SET; then
     AC_MSG_ERROR( Do NOT specify both --with-wasatch3p and --enable-wasatch_3p )
  fi
  if test "$with_spatialops" != NOT_SET; then
     AC_MSG_ERROR( Do NOT specify both --with-spatialops and --enable-wasatch_3p )
  fi
  if test "$with_exprlib" != NOT_SET; then
     AC_MSG_ERROR( Do NOT specify both --with-exprlib and --enable-wasatch_3p )
  fi
  if test "$with_tabprops" != NOT_SET; then
     AC_MSG_ERROR( Do NOT specify both --with-tabprops and --enable-wasatch_3p )
  fi
  if test "$with_radprops" != NOT_SET; then
     AC_MSG_ERROR( Do NOT specify both --with-radprops and --enable-wasatch_3p )
  fi
  if test "$with_nscbc" != NOT_SET; then
     AC_MSG_ERROR( Do NOT specify both --with-nscbc and --enable-wasatch_3p )
  fi

  if test "$build_wasatch_3p" != "yes"; then
    PIECES=$(echo $build_wasatch_3p | tr "," "\n")
    for piece in $PIECES; do
      if test $piece != "spatialops" -a $piece != "exprlib" -a $piece != "tabprops" -a $piece != "radprops" -a $piece != "nscbc"; then
        AC_MSG_ERROR( [build_wasatch_3p value of $piece is invalid - should be one of: spatialops, exprlib, tabprops, radprops, or nscbc.] )
      else
        case $piece in
          spatialops)
            with_spatialops=`$sci_pwd`/Wasatch3P/install/SpatialOps
          ;;
          exprlib)
            with_exprlib=`$sci_pwd`/Wasatch3P/install/ExprLib
          ;;
          tabprops)
            with_tabprops=`$sci_pwd`/Wasatch3P/install/TabProps
          ;;
          radprops)
            with_radprops=`$sci_pwd`/Wasatch3P/install/RadProps
          ;;
          nscbc)
            with_nscbc=`$sci_pwd`/Wasatch3P/install/NSCBC
          ;;
        esac
      fi
    done
  else
    with_spatialops=`$sci_pwd`/Wasatch3P/install/SpatialOps
    with_exprlib=`$sci_pwd`/Wasatch3P/install/ExprLib
    with_tabprops=`$sci_pwd`/Wasatch3P/install/TabProps
    with_radprops=`$sci_pwd`/Wasatch3P/install/RadProps
    with_nscbc=`$sci_pwd`/Wasatch3P/install/NSCBC
  fi

  mkdir -p Wasatch3P

  if test $build_wasatch_3p = "yes"; then
    components="all"
  else
    components=$build_wasatch_3p
  fi

  $path_to_src/build_scripts/build_wasatch_3p.sh `$sci_pwd` $components "$BOOST_LIBRARY_DIR" "$BOOST_INCLUDE_DIR" "$debug" "$IS_STATIC_BUILD" "$sci_required_cuda"

  if test $? != 0; then
     AC_MSG_ERROR( Failed to build Wasatch 3p libraries... )
  fi

fi

##  --  Handle with-wasatch3p --------------------------------

if test "$with_wasatch3p" != NOT_SET; then

  if test "$with_spatialops" != NOT_SET -o "$with_exprlib" != NOT_SET -o "$with_tabprops" != NOT_SET -o "$with_radprops" != NOT_SET -o "$with_nscbc" != NOT_SET; then
    AC_MSG_ERROR([Do not use --with-wasatch3p with any of: --with-spatialops, --with-exprlib,--with-tabprops, --with-radprops, or --with-nscbc.])
  fi

  AC_MSG_NOTICE(Using the Wastach 3p libs in: $with_wasatch3p)
  with_spatialops=$with_wasatch3p/SpatialOps
  with_exprlib=$with_wasatch3p/ExprLib
  with_tabprops=$with_wasatch3p/TabProps
  with_radprops=$with_wasatch3p/RadProps
  with_nscbc=$with_wasatch3p/NSCBC
fi

##  --  search for SpatialOps Library ------------------------

if test "$with_spatialops" != NOT_SET; then
  if test "$HAVE_BOOST" != "yes"; then
    AC_MSG_ERROR( --with-spatialops=DIR requires Boost. )
  fi

  AC_MSG_CHECKING(for SpatialOps files (For now we are just verifying the existence of lib and .h files)) 

  SPOPS_INCLUDE_DIR="include/spatialops"

  # Determine if W3P has CUDA...
  grep -q "#define ENABLE_CUDA" $with_spatialops/$SPOPS_INCLUDE_DIR/SpatialOpsConfigure.h
  if test $? = 0; then
    # W3P has cuda, verify that Uintah has CUDA too...
    if test "$HAVE_CUDA" != "yes"; then
      AC_MSG_ERROR( ['$with_spatialops' built with CUDA, but Uintah is not...] )
    fi
  else
    # W3P does not have CUDA, verify that Uintah does NOT have CUDA too...
    if test "$HAVE_CUDA" = "yes"; then
      AC_MSG_ERROR( ['$with_spatialops' not built with CUDA, but Uintah is...] )
    fi
  fi

  # Check includes
  for file in Nebo.h OperatorDatabase.h \
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

  AC_MSG_RESULT(yes)
  DEF_SPATIALOPS="#define HAVE_SPATIALOPS 1"
  INC_SPATIALOPS_H="-I$with_spatialops/include"
  SPATIALOPS_LIB_DIR_FLAG="-L$with_spatialops/lib/spatialops"

  SPATIALOPS_LIB_FLAG="-lspatialops"

  HAVE_SPATIALOPS="yes"
fi
AC_SUBST(HAVE_SPATIALOPS)
AC_SUBST(DEF_SPATIALOPS)
AC_SUBST(INC_SPATIALOPS_H)
AC_SUBST(SPATIALOPS_LIB_FLAG)
AC_SUBST(SPATIALOPS_LIB_DIR_FLAG)

##  --  search for Exprlib Library ------------------------

if test "$with_exprlib" != NOT_SET; then
  if test "$HAVE_SPATIALOPS" != "yes"; then
    AC_MSG_ERROR( --with-exprlib=DIR requires SpatialOps. )
  fi

  AC_MSG_CHECKING(for Exprlib files (For now we are just verifying the existence of lib and .h files)) 
  if test ! -d $with_exprlib; then
    echo ""
    AC_MSG_ERROR( '$with_exprlib' directory does not exist. )
  fi
  SPOPS_INCLUDE_DIR="include/expression"
  # Check includes
  for file in ExprDeps.h ExprLib.h Context.h Expression.h ExpressionID.h FieldManagerList.h \
              ExpressionBase.h ExpressionFactory.h ExpressionTree.h FieldManager.h FieldManagerBase.h \
              Functions.h Tag.h TransportEquation.h uintah; do
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

if test "$with_tabprops" != NOT_SET; then

  AC_MSG_CHECKING(for TabProps files (For now we are just verifying the existence of lib and .h files)) 
  for file in include/tabprops/StateTable.h lib/tabprops/libtabprops.a lib/tabprops/libinterp.a; do
    if test ! -f $with_tabprops/$file; then
      AC_MSG_ERROR( '$with_tabprops/$file' does not exist. )
    fi
  done

  AC_MSG_RESULT(yes)

  DEF_TABPROPS="#define HAVE_TABPROPS 1"
  INC_TABPROPS_H="-I$with_tabprops/include"
  TABPROPS_LIB_DIR_FLAG="-L$with_tabprops/lib/tabprops"
  TABPROPS_LIB_FLAG="-ltabprops -linterp"
  HAVE_TABPROPS="yes"
fi

AC_SUBST(HAVE_TABPROPS)
AC_SUBST(DEF_TABPROPS)
AC_SUBST(INC_TABPROPS_H)
AC_SUBST(TABPROPS_LIB_FLAG)
AC_SUBST(TABPROPS_LIB_DIR_FLAG)

##  --  search for RadProps ------------------------

if test "$with_radprops" != NOT_SET; then

  AC_MSG_CHECKING(for RadProps files (For now we are just verifying the existence of lib and .h files)) 
  for file in include/radprops/AbsCoeffGas.h include/radprops/Particles.h include/radprops/RadiativeSpecies.h lib/radprops/libradprops.a; do
    if test ! -f $with_radprops/$file; then
      AC_MSG_ERROR( '$with_radprops/$file' does not exist. )
    fi
  done

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

##  --  search for Wasatch 3P NSCBC Library ------------------------

if test "$with_nscbc" != NOT_SET; then

  AC_MSG_CHECKING(for Wasatch 3P NSCBC files (Header only - verifying .h files)) 
  for file in include/nscbc/CharacteristicBCBuilder.h include/nscbc/NSCBCToolsAndDefs.h include/nscbc/SpeedOfSound.h include/nscbc/TagManager.h; do
    if test ! -f $with_nscbc/$file; then
      AC_MSG_ERROR( '$with_nscbc/$file' does not exist. )
    fi
  done

  AC_MSG_RESULT(yes)

  DEF_NSCBC="#define HAVE_NSCBC 1"
  INC_NSCBC_H="-I$with_nscbc/include"
  HAVE_NSCBC="yes"
fi

AC_SUBST(HAVE_NSCBC)
AC_SUBST(DEF_NSCBC)
AC_SUBST(INC_NSCBC_H)

##  --  search for PoKiTT Library ------------------------

if test "$with_pokitt" != NOT_SET; then

  AC_MSG_CHECKING(for Pokitt files (For now we are just verifying the existence of lib and .h files)) 
  POKITT_INCLUDE_DIR="include/pokitt"
  # Check includes
  for file in CanteraObjects.h  MixtureFractionExpr.h PoKiTTVersion.h   SpeciesN.h thermo \
              MixtureFraction.h MixtureMolWeight.h    SpecFromMixfrac.h kinetics   transport; do
    if test ! -e $with_pokitt/$POKITT_INCLUDE_DIR/$file; then
      echo ""
      AC_MSG_ERROR( '$with_pokitt/$POKITT_INCLUDE_DIR/$file' does not exist. )
    fi
  done
  # Check lib
  the_lib=$with_pokitt/lib/pokitt/libpokitt.a
  if test ! -f $the_lib; then
    echo ""
    AC_MSG_ERROR( '$the_lib' does not exist. )
  fi

  ## Now check for Cantera based on POKITT installation...
  cantera_info=$with_pokitt/share/PoKiTTConfig.cmake
  if test ! -f $cantera_info; then
    echo ""
    AC_MSG_ERROR( '$cantera_info' does not exist. )
  fi

  # The CANTERA_LIB_FLAG is the whole path to (and including) the .a file.
  CANTERA_LIB_FLAG=`grep Cantera_LIBRARIES   $cantera_info | cut -f3 -d" "`
  INC_CANTERA_H=`grep Cantera_INCLUDE_DIR $cantera_info | cut -f3 -d" "`

  if test ! -d "$INC_CANTERA_H"; then
    echo ""
    AC_MSG_ERROR(Cantera include dir ($INC_CANTERA_H) does not exist.)
  fi

  # Add -I to include var:
  INC_CANTERA_H=-I$INC_CANTERA_H

  DEF_POKITT="#define HAVE_POKITT 1"
  INC_POKITT_H="-I$with_pokitt/include $INC_CANTERA_H"

  POKITT_LIB_DIR_FLAG="-L$with_pokitt/lib/pokitt"
  POKITT_LIB_FLAG="-lpokitt $CANTERA_LIB_FLAG"

  HAVE_POKITT="yes"
  AC_MSG_RESULT(yes)
fi

AC_SUBST(HAVE_POKITT)
AC_SUBST(DEF_POKITT)
AC_SUBST(INC_POKITT_H)
AC_SUBST(POKITT_LIB_FLAG)
AC_SUBST(POKITT_LIB_DIR_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/wasatch_testdefs.h"

##  --  Make sure Wasatch 3P is found if WASATCH or ARCHES is requested -------

if test "$BUILD_WASATCH" = "yes"; then
  if test "$HAVE_BOOST"      != "yes" -o \
          "$HAVE_SPATIALOPS" != "yes" -o \
          "$HAVE_EXPRLIB"    != "yes" -o \
          "$HAVE_TABPROPS"   != "yes" -o \
          "$HAVE_RADPROPS"   != "yes" -o \
          "$HAVE_NSCBC"      != "yes"; then
    AC_MSG_ERROR( [You enabled the Wasatch component but do not have all of the prereqs: Boost ($HAVE_BOOST), TabProps ($HAVE_TABPROPS), RadProps ($HAVE_RADPROPS), SpatialOps ($HAVE_SPATIALOPS), ExprLib ($HAVE_EXPRLIB), and NSCBC ($HAVE_NSCBC).] )
  fi
fi

if test "$BUILD_ARCHES" = "yes"; then
  if test "$HAVE_BOOST"      != "yes" -o \
          "$HAVE_TABPROPS"   != "yes" -o \
          "$HAVE_RADPROPS"   != "yes"; then
    AC_MSG_ERROR( [You enabled the Arches component but do not have all of the prereqs: Boost ($HAVE_BOOST), TabProps ($HAVE_TABPROPS), or RadProps ($HAVE_RADPROPS).] )
  fi
fi

if test "$BUILD_WASATCH" = "yes" -o "$BUILD_ARCHES" = "yes"; then
  if test "$with_hypre" = NOT_SET -a "$with_hypre_include" = NOT_SET; then
    AC_MSG_WARN( [ Requiring Hypre due to Arches or Wasatch...] )
    with_hypre=yes
  fi
fi

##  --  search for hypre stuff  ------------------------------------------

_sci_lib=
_sci_inc=
_sci_optional=optional

# Handle --without option
if test "$with_hypre" = "no"; then
   AC_MSG_WARN(Hypre is turned off.)
else
  if test "$with_hypre" = "built-in"; then
     HAVE_HYPRE="yes"
     DEF_HYPRE="#define HAVE_HYPRE 1"
     AC_MSG_WARN(Relying on a built-in version of Hypre)
  else
     if test "$with_hypre" != NOT_SET ; then
       if test "$with_hypre_include" != NOT_SET -o "$with_hypre_lib" != NOT_SET ; then
         AC_MSG_ERROR( You cannot use --with-hypre and --with-hypre-include/--with-hypre-library. )
       elif test "$with_hypre" != "yes"; then
         _sci_lib="$with_hypre/lib"
         _sci_inc="$with_hypre/include"
         _sci_optional=specific
       fi
     elif test "$with_hypre_lib" != NOT_SET -o "$with_hypre_include" != NOT_SET; then
       if test "$with_hypre_lib" = NOT_SET -o "$with_hypre_include" = NOT_SET; then
         AC_MSG_ERROR( You must specify both --with-hypre-include and --with-hypre-library. )
       else
         _sci_lib="$with_hypre_lib"
         _sci_inc="$with_hypre_include"
         with_hypre="yes"
         _sci_optional=specific
       fi

	# Try to avoid accidentally allowing system libraries to shadow the specific libraries the user has requested. 
        if test "$USING_GCC" = "yes" -o "$USING_ICC" = "yes"; then
           ESCHEW_SYS_LIBS=-nodefaultlibs
        else
           ESCHEW_SYS_LIBS=
        fi
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
            SCI_TRY_LINK(HYPRE,HYPRE libraries and headers - 1,HYPRE.h,$_sci_inc,
                        HYPRE HYPRE_utilities HYPRE_struct_mv HYPRE_struct_ls HYPRE_sstruct_mv HYPRE_sstruct_ls HYPRE_parcsr_ls HYPRE_krylov HYPRE_parcsr_mv,
                        $_sci_lib,
                        $MPI_LIB_DIR_FLAG $MPI_LIB_FLAG $MALLOC_TRACE_LIB_DIR_FLAG $MALLOC_TRACE_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG $M_LIB_DIR_FLAG $M_LIB_FLAG $DL_LIB_FLAG $THREAD_LIB_FLAG $ESCHEW_SYS_LIBS,
                        ,
                        optional)

     if test "$HAVE_HYPRE" != "yes"; then
           SCI_TRY_LINK(HYPRE,HYPRE libraries and headers - 2,HYPRE.h,$_sci_inc,
                           HYPRE,
                           $_sci_lib,
                           $MPI_LIB_DIR_FLAG $MPI_LIB_FLAG $MALLOC_TRACE_LIB_DIR_FLAG $MALLOC_TRACE_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $F_LIB_DIR_FLAG $F_LIB_FLAG $M_LIB_DIR_FLAG $M_LIB_FLAG $DL_LIB_FLAG $THREAD_LIB_FLAG,
                           ,
                           optional)

     fi
     AC_LANG([C])

     if test "$HAVE_HYPRE" = "yes" ; then
        DEF_HYPRE="#define HAVE_HYPRE 1"
     fi

     if test "$with_hypre" != NOT_SET && test ! "$DEF_HYPRE"; then
        echo
        AC_MSG_ERROR(One or more of the HYPRE components is missing.)
     fi
  fi
fi

AC_SUBST(HAVE_HYPRE)
AC_SUBST(DEF_HYPRE)
AC_SUBST(INC_HYPRE_H)
AC_SUBST(HYPRE_LIB_FLAG)
AC_SUBST(HYPRE_LIB_DIR_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/hypre_testdefs.h"

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

    if test -d StandAlone -a "$file" = "include/sci_defs/wasatch_defs.h"; then
       echo "WARNING: Wasatch 3P dependent .o files are probably out of date... removing..."
       $path_to_src/build_scripts/repair.sh -quiet SpatialOps
       $path_to_src/build_scripts/repair.sh -quiet ExprLib
       $path_to_src/build_scripts/repair.sh -quiet TabProps
       $path_to_src/build_scripts/repair.sh -quiet RadProps
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
