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
##  It used to be the case that the SCIRun make system tried to run
##  autconf automatically if configure.ac had changed.  This caused a
##  number of problems and has been removed.  Therefore the order
##  of committing configure and configure.ac no longer matters.  I
##  suggest committing them both at the same time now.
##    - Dav
##
##  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##  --------------------------------------------------------------------
##  ------------------------  initialize  ------------------------------
##  --------------------------------------------------------------------

##  initialize configure 
AC_INIT([SCIRun],[1.24],[scirun-users@sci.utah.edu])

##  require autoconf version 2.59 
##  type "autoconf --version" to find out what version you have.
AC_PREREQ(2.59)

AC_COPYRIGHT(
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
)

##
## Any error/warning/note/remark messages that need to be turned off
## globally should be appended to this variable.  This should only be
## done if the message is pervasive and has been determined to be
## benign.  WOFF_NUMBERS should be a 'space' delimited list of numbers.
## Eg: '123 124 155'
##
WOFF_NUMBERS=

## Clean up files that could be left around if configure was 
## interrupted (by control-c, ^c) because I do not know how to add
## cleanup to the builtin AC function that uses the 'trap' command.
rm -rf ii_files sedscript

AC_CONFIG_SRCDIR(configVars.mk.in)

##  set the location of config.guess, config.sub, and install.sh
AC_CONFIG_AUX_DIR(scripts)

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

#Get the host now since it is needed for the Package Checks.
AC_CANONICAL_HOST

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
   sci_required_blas=yes],
  [with_blas=""])
SCI_ARG_WITH(lapack,
  AC_HELP_STRING(--with-lapack=DIR, 
                 Use the lapack installation in DIR),
  [with_lapack="$withval"
   sci_required_lapack=yes],
  [with_lapack=""])
SCI_ARG_WITH(fortran,
  AC_HELP_STRING(--with-fortran=DIR, 
                 Use the fortran (eg: g2c) installation in DIR/lib),
  [with_fortran="$withval"
   sci_required_fortran=yes],
  [with_fortran=""])
SCI_ARG_WITH(ptolemyII,
  AC_HELP_STRING(--with-ptolemyII=DIR, 
                 Use the Ptolemy II (eg: ptII4.0.1) installation in DIR (Ptolemy package only)),
  [with_ptolemyII="$withval"
   sci_required_ptolemyII=yes],
  [with_ptolemyII=""])
SCI_ARG_WITH(java,
  AC_HELP_STRING(--with-java=DIR,
                 Use the Java distribution in DIR (Ptolemy package only)),
  [with_java="$withval"
   sci_required_java=yes],
  [with_java=""])
SCI_ARG_WITH(mpi,
  AC_HELP_STRING(--with-mpi=DIR,
                 Use the mpi installation in DIR),
  [with_mpi="$withval"
   sci_required_mpi=yes],
  [with_mpi=""])
SCI_ARG_WITH(mpiuni,
  AC_HELP_STRING(--with-mpiuni=DIR,
                 Use the mpiuni installation in DIR),
  [with_mpiuni="$withval"
   sci_required_mpiuni=yes],
  [with_mpiuni=""])
SCI_ARG_WITH(hypre,
  AC_HELP_STRING(--with-hypre=DIR,
                 Use the hypre installation in DIR),
  [with_hypre="$withval"
   sci_required_hypre=yes],
  [with_hypre=""])
SCI_ARG_WITH(netsolve,
  AC_HELP_STRING(--with-netsolve=DIR,
                 Use the netsolve installation in DIR),
  [with_netsolve="$withval"
   sci_required_netsolve=yes],
  [with_netsolve=""])
SCI_ARG_WITH(petsc,
  AC_HELP_STRING(--with-petsc=DIR,
                 Use the petsc installation in DIR),
  [with_petsc="$withval"
   sci_required_petsc=yes],
  [with_petsc=""])
SCI_ARG_WITH(png,
  AC_HELP_STRING(--with-png=DIR,
                 Use the png installation in DIR),
  [with_png="$withval"
   sci_required_png=yes],
  [with_png=""])
SCI_ARG_WITH(camal,
  AC_HELP_STRING(--with-camal=DIR,
                 Use the CMLTET installation in DIR),
  [with_camal="$withval"
   sci_required_camal=yes],
  [with_camal=""])
SCI_ARG_WITH(magick,
  AC_HELP_STRING(--with-magick=DIR,
                 Use the Image Magick installation in DIR),
  [with_magick="$withval"
   sci_required_magick=yes],
  [with_magick=""])
SCI_ARG_WITH(unipetsc,
  AC_HELP_STRING(--with-unipetsc=DIR,
                 Use the unipetsc installation in DIR),
  [with_unipetsc="$withval"
   sci_required_unipetsc=yes],
  [with_unipetsc=""])
SCI_ARG_WITH(tau,
  AC_HELP_STRING(--with-tau=<tau-makefile>,
                 Specify a Tau Makefile),
  [with_tau="$withval"
   sci_required_tau=yes],
  [with_tau=""])
SCI_ARG_WITH(unp,
  AC_HELP_STRING(--with-unp=DIR,
                 Use the unp installation in DIR),
  [with_unp="$withval"
   sci_required_unp=yes],
  [with_unp=""])
SCI_ARG_WITH(globus,
  AC_HELP_STRING(--with-globus=DIR,
                 Use the globus installation in DIR),
  [with_globus="$withval"
   sci_required_globus=yes
   sci_required_ssl=yes
   sci_required_crypto=yes],
  [with_globus=""])
SCI_ARG_WITH(crypto,
  AC_HELP_STRING(--with-crypto=DIR,
                 Use the crypto installation in DIR (only for use with globus)),
  [with_crypto="$withval"
   sci_required_crypto=yes],
  [with_crypto=""])
SCI_ARG_WITH(ssl,
  AC_HELP_STRING(--with-ssl=DIR,
                 Use the ssl installation in DIR (only for use with globus)),
  [with_ssl="$withval"
   sci_required_ssl=yes],
  [with_ssl=""])
SCI_ARG_WITH(insight,
  AC_HELP_STRING(--with-insight=DIR,
	         Use the Insight installation in DIR),
  [with_insight="$withval"
   sci_required_insight=yes],
  [with_insight=""])
SCI_ARG_WITH(qt,
  AC_HELP_STRING(--with-qt=DIR,
                 Use the Qt installation in DIR),
  [with_qt="$withval"
   sci_required_qt=yes],
  [with_qt=""])
SCI_ARG_WITH(ruby,
  AC_HELP_STRING(--with-ruby=DIR,
                 Use the ruby installation in DIR where DIR points to the --prefix option specified when installing ruby.  A standard ruby install defaults to --prefix=/usr/local),
  [with_ruby="$withval"
   sci_required_ruby=yes],
  [with_ruby=""])
SCI_ARG_WITH(tao,
  AC_HELP_STRING(--with-tao=DIR,
                 Use the tao corba installation in DIR where DIR points to the --prefix option specified when installing tao.),
  [with_tao="$withval"
   sci_required_tao=yes],
  [with_tao=""])
SCI_ARG_WITH(vtk,
  AC_HELP_STRING(--with-vtk=DIR,
                 Build the VTK components. Use VTK build in DIR.  DIR should be the directory specified in the VTK CMake variable CMAKE_INSTALL_PREFIX, which is /usr/local by default),
  [with_vtk="$withval"
   sci_required_vtk=yes],
  [with_vtk=""])
SCI_ARG_WITH(opengl,
  AC_HELP_STRING(--with-opengl=DIR,
                 Use the OpenGL installation in DIR),
  [with_opengl="$withval"
   sci_required_opengl=yes],
  [with_opengl=""])
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

SCI_ARG_WITH(mdsplus,
  AC_HELP_STRING(--with-mdsplus=DIR,
                 Use the mdsplus installation in DIR),
  [with_mdsplus="$withval"
   sci_required_mdsplus=yes],
  [with_mdsplus=""])

SCI_ARG_WITH(hdf5,
  AC_HELP_STRING(--with-hdf5=DIR,
                 Use the hdf5 installation in DIR),
  [with_hdf5="$withval"		      # with_hdf5 is the directory var
   sci_required_hdf5=yes],		      # flag to check for the libs
  [with_hdf5=""])		      # undefinded otherwise

SCI_ARG_WITH(glut,
  AC_HELP_STRING(--with-glut=DIR,
                 Use the glut installation in DIR),
  [with_glut="$withval"
   sci_required_glut=yes],
  [with_glut=""])
SCI_ARG_WITH(glui,
  AC_HELP_STRING(--with-glui=DIR,
                 Use the glui installation in DIR),
  [with_glui="$withval"
   sci_required_glui=yes],
  [with_glui=""])
SCI_ARG_WITH(oogl,
  AC_HELP_STRING(--with-oogl=DIR,
                 Use the oogl installation in DIR),
  [with_oogl="$withval"
   sci_required_oogl=yes],
  [with_oogl=""])
SCI_ARG_WITH(glew,
  AC_HELP_STRING(--with-glew=DIR,
                 Use the GLEW installation in DIR),
  [with_glew="$withval"
   sci_required_glew=yes],
  [with_glew=""])
SCI_ARG_WITH(audio,
  AC_HELP_STRING(--with-audio=DIR,
                 Use the audio installation in DIR),
  [with_audio="$withval"
   sci_required_audio=yes],
  [with_audio=""])
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
SCI_ARG_WITH(uuid,
  AC_HELP_STRING([--with-uuid],
                 Use UUID tools on IRIX or from ext2fs development tools on GNU/linux (SCIRun2 only, default).  Installing the ext2fs package on GNU/Linux is recommended.),
  [sci_required_uuid=yes],
  [with_uuid=""])
SCI_ARG_WITH(boost,
  AC_HELP_STRING(--with-boost=DIR,
                 Use the Boost in DIR),
  [with_boost="$withval"
   sci_required_boost=yes],
  [with_boost=""])
SCI_ARG_WITH(loki,
  AC_HELP_STRING(--with-loki=DIR,
                 Use the Loki in DIR),
  [with_loki="$withval"
   sci_required_loki=yes],
  [with_loki=""])



SCI_ARG_ENABLE(advanced-volume-rendering,
  AC_HELP_STRING(--enable-advanced-volume-rendering,
                 Override default tests for this capability),
  [avr="$enableval"],
  [avr="test"])


SCI_ARG_ENABLE(advanced-volume-rendering,
  AC_HELP_STRING(--enable-advanced-volume-rendering,
                 Override default tests for this capability),
  [avr="$enableval"],
  [avr="test"])
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
  [sci_required_perfex=yes],
  [sci_required_perfex=no])
SCI_ARG_ENABLE(package,
  AC_HELP_STRING(--enable-package=<list>,
                 Use the SCIRun packages specified by list.
                 example: --enable-package='BioPSE Teem'),
  [package="$enableval"],
  [package=""])
SCI_ARG_ENABLE(dataflow,
  AC_HELP_STRING(--disable-dataflow,
                 do not build Dataflow),
  [sci_required_dataflow=no],
  [sci_required_dataflow=yes])
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
SCI_ARG_ENABLE(scirun2,
  AC_HELP_STRING(--enable-scirun2,
                 Build experimental SCIRun 2),
  [enable_scirun2="$enableval"
   [SCI_SET_PACKAGE_CHECKS(SCIRun2, $host)]
  ],
  [enable_scirun2=${enable_scirun2='no'}])
SCI_ARG_ENABLE(plume,
  AC_HELP_STRING(--enable-plume,
                 Build Plume (based on SCIRun 2)),
  [enable_plume="$enableval"
   [SCI_SET_PACKAGE_CHECKS(Plume, $host)]
   ],
  [enable_plume=${enable_plume='no'}])
SCI_ARG_WITH(tena,
  AC_HELP_STRING(--with-tena=DIR,
                 Use the TENA installation in DIR),
  [with_tena="$withval"
   sci_required_tena=yes],
  [with_tena=""])
SCI_ARG_WITH(om,
  AC_HELP_STRING(--with-om=DIR,
                 Use the TENA OM installation in DIR),
  [with_tena_om="$withval"
   sci_required_tena_om=yes],
  [with_tena_om=""])
SCI_ARG_ENABLE(rath,
  AC_HELP_STRING(--enable-rpath,
                 Use the default runtime library search path),
  [enable_rpath="$enableval"],
  [enable_rpath=${enable_rpath='yes'}])
SCI_ARG_ENABLE(chromium,
  AC_HELP_STRING(--enable-chromium,
                 Using Chromium so skip unsupported OGL calls),
  [enable_chromium="$enableval"],
  [enable_chromium=${enable_chromium='no'}])
SCI_ARG_ENABLE(ati-card-fix,
  AC_HELP_STRING(--enable-ati-card-fix,
                 Enable the ati car fix to repair bad polygons - Deprecated - use SCIRUN_DRAWARRAYS_DISABLE environment variable - see scirunrc file),
  [enable_ati_card_fix="$enableval"],
  [enable_ati_card_fix=${enable_ati_card_fix='no'}])
SCI_ARG_ENABLE(no-dynamic-cast,
  AC_HELP_STRING(--enable-no-dynamic-cast,
	         Disable dynamic casting, SGI compiler bug workaround.),
  [enable_no_dynamic_cast="$enableval"],
  [enable_no_dynamic_cast=${enable_no_dynamic_cast='no'}])


if test "$enable_rpath" = "no"; then
   AC_MSG_WARN("You have removed the rpath flag from some libraries.")
   AC_MSG_WARN("This may have unpredictable affects.")
   AC_MSG_WARN("Do not do this unless you really know what you are doing.")
fi

if test "$enable_chromium" = "yes"; then
   DEF_CHROMIUM="#define HAVE_CHROMIUM 1"
fi
AC_SUBST(DEF_CHROMIUM)
sci_defs_files="$sci_defs_files include/sci_defs/chromium_testdefs.h"

if test "$enable_ati_card_fix" = "yes"; then
   AC_MSG_WARN("The ATI card hack is no longer controlled by configure.")
   AC_MSG_WARN("You can turn it on/off by setting the environment variable")
   AC_MSG_WARN("SCIRUN_DRAWARRAYS_DISABLE to true.  This can be done either")
   AC_MSG_ERROR("in your environment or in your .scirunrc file.)
fi

if test "$enable_no_dynamic_cast" = "yes"; then
   DEF_NO_DYNAMIC_CAST="#define NO_DYNAMIC_CAST 1"
fi
AC_SUBST(DEF_NO_DYNAMIC_CAST)
sci_defs_files="$sci_defs_files include/sci_defs/dynamic_cast_testdefs.h"

##  --------------------------------------------------------------------
##  Save off the arguments to configure, being careful to save 
##  Package strings for later parsing.
##  --------------------------------------------------------------------

args=`echo "$*" | sed 's% %_%g'`
myargs=`echo $args | sed 's%\^% %g'`
# Setting IFS breaks standard autoconf tests.  
# It thinks we are cross compiling. Don't set it.
#IFS=' '

## Cut off the /configure from the configure command (eg: ../src/configure)
path_to_src=`echo $0 | sed 's/\/configure//'`

 
BUILD_DATAFLOW="$sci_required_dataflow"
if test "$sci_required_dataflow"; then
  DEF_BUILD_DATAFLOW="#define BUILD_DATAFLOW"
fi
AC_SUBST(DEF_BUILD_DATAFLOW)
AC_SUBST(BUILD_DATAFLOW)
sci_defs_files="$sci_defs_files include/sci_defs/dataflow_testdefs.h"


if test "$package" = "all"; then
  AC_MSG_RESULT(Enabling all the packages in your SCIRun/src/Packages directory.)
  _sci_tmp=`ls $path_to_src/Packages`
  package=`echo $_sci_tmp | sed 's/CVS //g'`
  AC_MSG_RESULT(Packages include $package)
fi
PACKAGE_DIRS=""
for i in $package; do
   PACKAGE_DIRS=$PACKAGE_DIRS" Packages/$i"
   # set the package dependent checks
   SCI_SET_PACKAGE_CHECKS($i, $host)
done

if test "$pkg_BioPSE" = "yes" || test "$pkg_Insight" = "yes" ||
   test "$pgk_MatlabInterface" = "yes" || test "$pkg_DataIO" = "yes" ||
   test "$pkg_Fusion" = "yes"; then
   if test "$pkg_Teem" != "yes"; then
      AC_MSG_WARN(User did not enable the Teem package but it is required by)
      AC_MSG_WARN(an enabled package(s)... adding it to the list of requested packages.)
      PACKAGE_DIRS=$PACKAGE_DIRS" Packages/Teem"
      # set the package dependent checks
      SCI_SET_PACKAGE_CHECKS(Teem, $host)
      package="$package Teem"
   fi
fi

if test "$pkg_Fusion" = "yes"; then
  if test "$pkg_DataIO" != "yes"; then
      AC_MSG_WARN(User did not enable the DataIO package but it is required by)
      AC_MSG_WARN(the Fusion package... adding it to the list of requested packages.)
      PACKAGE_DIRS=$PACKAGE_DIRS" Packages/DataIO"
      # set the package dependent checks
      SCI_SET_PACKAGE_CHECKS(DataIO, $host)
      package="$package DataIO"
  fi
fi

if test "$pkg_Ptolemy" = "yes"; then
  # check versions?
  # check --with-ptolemyII
  if test -z "$with_ptolemyII"; then
    AC_MSG_WARN(Ptolemy II is required by the Ptolemy package.)
    AC_MSG_ERROR(Use the --with-ptolemyII=DIR option to set the installation directory.)
  fi
  SCI_SET_PACKAGE_CHECKS(Ptolemy, $host)
fi

# Make sure the Package directories exist.
#   Do an 'ls' of all files in the Packages directory and then grep for the
#   package in this list.  (We used to just do a "test -d $package", but on
#   a case-insensitive Mac file system, this fails.)

packages_list=`ls $path_to_src/Packages`
for i in $package; do
   found_package=`echo $packages_list | grep $i`
   if test -z "$found_package"; then
      if test "$i" = "TEEM"; then
         AC_MSG_WARN(Try using 'Teem' instead of 'TEEM')
      fi
      AC_MSG_ERROR(Package '$i' does not exist (in $path_to_src/Packages/)!)
   fi
done

if test "$sci_required_unipetsc" = "yes"; then
   sci_required_blas=yes
   sci_required_lapack=yes
fi

# Set the collaborative vars.
if test "$pkg_CollabVis" = "yes"; then
  DEF_COLLAB_VIS="#define HAVE_COLLAB_VIS 1"
  HAVE_COLLAB_VIS="yes"
fi

AC_SUBST(DEF_COLLAB_VIS)
AC_SUBST(HAVE_COLLAB_VIS)
sci_defs_files="$sci_defs_files include/sci_defs/collab_vis_testdefs.h"

LOAD_PACKAGE="SCIRun"
for i in $package; do
 LOAD_PACKAGE=$LOAD_PACKAGE,"$i"
done
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

## -- Can't use uni-petsc and Uintah (as Uintah needs full mpi petsc).

if test "$pkg_Uintah" = "yes" && test "$with_unipetsc"; then
   AC_MSG_ERROR(You have requested both Uintah and Uni-Petsc.  However Uintah requires MPI petsc.)
fi

##  --------------------------------------------------------------------
##  -----  determine host type, processor count and OS version  --------
##  --------------------------------------------------------------------

echo 
IS_OSX="no"
IS_WIN="no"
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
    SCI_CHECK_OS_VERSION(7.1,NEED_OSX_HACK=no,NEED_OSX_HACK=yes)
    NUM_CPUS=2
    IS_OSX="yes"
    ;;
  *-aix*)
    NUM_CPUS=4
    ;;
  *-mingw*)
    NUM_CPUS=2
    IS_WIN="yes"
    ;;
  *-pc*)
    echo
    AC_MSG_ERROR(Currently Windows only handles mingw.  See install instructions)
    ;;
  *)
    echo
    AC_MSG_WARN( !!!!!! This system is untested - You are on your own !!!!!!)
    echo
    NUM_CPUS="1"
    ;;
esac

AC_SUBST(IS_OSX)
AC_SUBST(IS_WIN)
sci_defs_files="$sci_defs_files include/sci_defs/osx_testdefs.h"

AC_MSG_CHECKING(physical processor count)
if test -z "$NUM_CPUS"; then
  NUM_CPUS="1"
fi
AC_MSG_RESULT($NUM_CPUS)


##  ----------  determine if this is a 32 or 64 bit machine
AC_LANG([C])

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

# the next two lines were moved to a later stage to allow us to change this default based on --enable-32/64bit
#AC_SUBST(NBITS)
#sci_defs_files="$sci_defs_files include/sci_defs/bits_testdefs.h"

## --------------

AC_LANG([C++])

if test "$CC" = ""; then
  # Apparently AC_PROG_CC will set CFLAGS to "-O2 -g" if you don't specify
  # its value first, so...
  CFLAGS=$CFLAGS

  AC_PROG_CC(cc gcc)

  case "$host" in 
     *-irix*)
        AC_MSG_WARN(Configure script only allows SGI compilers on IRIX.)
        AC_MSG_WARN(If this is a problem email scirun-users@sci.utah.edu.)

        if test $CC != cc; then
           AC_MSG_ERROR(Configure did not find the 'cc' SGI C compiler.)
        fi
	;;
     *)
        #qwerty
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


if test "$F77" = ""; then
  AC_PROG_F77()
  AC_MSG_WARN(Fortran compiler not specified with 'F77' environment variable. Using default... $F77)
fi

# Most fortran compilers need an "_" for C function names.
# Some have strange underscore needs.  This flag is used in the
# Packages/Uintah/tools/fspec.pl.in file to generate the correct
# cname of the fortran function.
FORTRAN_UNDERSCORE=end

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
	   AC_MSG_WARN(sci_malloc does not work under mac osx. use -enable-sci-malloc=force)
	   AC_MSG_WARN(sci-malloc disabled)
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

ASSERTION_LEVEL="$enable_assertion_level"
THROW_LEVEL="$enable_throw_level"
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
  DEF_MALLOC="#define DISABLE_SCI_MALLOC 1"
fi

if test "$enable_scinew_line_numbers" = "yes"; then
  DEF_MALLOC_LINENUMBERS="#define USE_TAG_LINENUM 1"
else
  DEF_MALLOC_LINENUMBERS=""
fi


AC_SUBST(SO_OR_ARCHIVE_EXTENSION)
AC_SUBST(NOPRINTDIR)
AC_SUBST(PACKAGE_DIRS)
AC_SUBST(LOAD_PACKAGE)
AC_SUBST(EXENAME)
AC_SUBST(LARGESOS)
AC_SUBST(TIME_IMPL)

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

AC_MSG_CHECKING(whether debug was enabled)
OGL_ERR_MACRO=""
if test "$debug" = "no"; then
  AC_MSG_RESULT(no)
elif test "$debug" = "yes"; then
  OGL_ERR_MACRO="#define OGL_DBG 1"
  extra_cflags="-g ${extra_cflags}"
  AC_MSG_RESULT(yes (-g) )
else
  OGL_ERR_MACRO="#define OGL_DBG 1"
  extra_cflags="$debug ${extra_cflags}"
  AC_MSG_RESULT(yes ($debug) )
fi

AC_SUBST(OGL_ERR_MACRO)

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

if test "$sci_required_perfex" = "yes"; then
  extra_cflags="-DUSE_PERFEX_COUNTERS ${extra_cflags}"
fi

if test ! -e "`type $CC | sed 's%^.* %%'`"; then
  SCI_MSG_ERROR(Could not find the specified C compiler ($CC))
fi
if test ! -e "`type $CXX | sed 's%^.* %%'`"; then
  SCI_MSG_ERROR(Could not find the specified C++ compiler (CC))
fi

if test $sci_required_fortran = "yes"; then
  if test ! -e "`type $F77 | sed 's%^.* %%'`"; then
    SCI_MSG_ERROR(Could not find the specified Fortran compiler ($F77))
  fi
fi

# This flag is for the windows compiler - so it can generate correct
# SHARE flags.
WIN_DLLFLAGS=""

case $host in 
  *-irix*)
    ac_objext=o

    # Check for the use of the 7.4 compiler (so that we can turn off
    # some extra warnings.)
    SCI_CHECK_VERSION($CXX,-version,7.4,USING_74=yes,)
    if test "$USING_74" = "yes"; then
      WOFF_NUMBERS="$WOFF_NUMBERS 3625 3649"
    fi

    SCI_CHECK_VERSION($CC,-version,7.3.1.1,,
                      echo configure: error: wrong $CC version.; exit)
    SCI_CHECK_VERSION($CXX,-version,7.3.1.1,,
                      echo configure: error: wrong $CXX version.; exit)
    if test $sci_required_fortran = "yes"; then
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
    if test "$USING_74" = "yes"; then
        WOFF=" -woff 3625,3303,1047,1209,1110 -Wl,-woff,84 -woff 1342"
    else
        WOFF=" -woff 3303,1047,1209,1110 -Wl,-woff,84 -woff 1342"
    fi
    CPPFLAGS=" $WOFF -LANG:std"

    if test "$enable_64bit" = "yes"; then
      if test $NBITS = "32"; then
        BINARYFLAGS="-64 -mips4"
      fi
      NBITS=64
    else
      NBITS=32
      BINARYFLAGS="-n32 -mips4"
    fi

    FFLAGS="$BINARYFLAGS $extra_cflags"
    CFLAGS="-fullwarn -xansi -LANG:std $BINARYFLAGS $extra_cflags"
    CXXFLAGS="$CFLAGS"
    ASFLAGS="$BINARYFLAGS"
    SOFLAGS="-shared $BINARYFLAGS -LANG:std -no_unresolved -J\$(MAKE_PARALLELISM) -update_registry \$(LIBDIR)/so_locations $(TLINK) ${SOFLAGS}"
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
    FFLAGS="$extra_cflags $FFLAGS"
    CFLAGS="$extra_cflags -D__USE_STD_IOSTREAM $CFLAGS"
    CXXFLAGS="-model ansi $extra_cflags -D__USE_STD_IOSTREAM $CXXFLAGS"
    if test "$threads" != "none"; then
      CFLAGS="$CFLAGS -pthread"
      CXXFLAGS="$CXXFLAGS -pthread"
    fi
    SOFLAGS="-shared -warning_unresolved -update_registry \$(LIBDIR)/so_locations"
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

         # Tell the compiler that the executable can use a
         # maximum of 1280MB (5*256MB) of memory per processor
         # rather than the default of 256MB.
         LDFLAGS="-bmaxdata:0x50000000 $LDFLAGS"

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
     SCI_CHECK_VERSION($CC,--version,4.0.0,GCC_VERSION=4.0,)

    if test "$GCC_VERSION" = "4.0"; then
       CFLAGS="$CFLAGS $extra_cflags -fno-common -dynamic -Wno-long-double "
    else
       CFLAGS="$CFLAGS $extra_cflags -fno-common -dynamic -Wno-long-double -fcoalesce-templates"
    fi   
    CXXFLAGS="$CXXFLAGS $CFLAGS"

    if test $NEED_OSX_HACK = yes; then
	SOFLAGS="-dynamiclib -init __ZNSt8ios_base4InitC4Ev \${LIBDIR}/../osx/osx_init.o  ${SOFLAGS}"
        DEF_NEED_OSX_HACK="#define NEED_OSX_HACK 1"
        DEF_THREAD="#define SCI_PTHREAD 1"
    else
	SOFLAGS="-dynamiclib"
    fi
    case $F77 in
      *g77*)
         # extra_fflags has array checking if in debug mode.
         FFLAGS="$FFLAGS $extra_cflags $extra_fflags"

         # put '_' at end and two '_'s if there is a '_' in name.
         # This flag is used by the genfspec.pl script in Uintah (generated
         # from fspec.pl.in).
         FORTRAN_UNDERSCORE=linux
         ;;
      *gfortran*)
         FFLAGS="$FFLAGS $extra_cflags"   
         FORTRAN_UNDERSCORE=linux
         ;;        
      *)
         FFLAGS="$FFLAGS $extra_cflags"
         ;;
    esac
    AC_SUBST(NEED_OSX_HACK)
    AC_SUBST(DEF_NEED_OSX_HACK)
    ;;

  *-pc-mingw*)
    ac_objext=o
    CFLAGS=" -Wall"
    CXXFLAGS=" -Wall"
    WIN_DLLFLAGS=' -DBUILD_$(subst /,_,$(patsubst %/,%,$(dir $@)))'
    SCI_CHECK_VERSION($CC,--version,2.95.3,,
      echo configure: error: wrong $CC version.; exit)
    if test $sci_required_fortran = "yes"; then
      SCI_CHECK_VERSION($F77,--version,0.5.25,,
        echo configure: error: wrong $F77 version.; exit)
    fi
    SCI_CHECK_VERSION($CXX,--version,2.95.3,,
      echo configure: error: wrong $CXX version.; exit)
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
      *gfortran*)
         FFLAGS="$FFLAGS $extra_cflags"   
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
  *)
    # linux
    ac_objext=o

    if test "$CC" = "icc"; then
      CFLAGS=" -w1"
      CXXFLAGS=" -w1"
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
    if test "$enable_32bit" = "yes" -a $NBITS=64; then
       NBITS=32
       CFLAGS="-m32 $CFLAGS"
       CXXFLAGS="-m32 $CXXFLAGS"
       AC_MSG_RESULT(-m32)
    else 
      if test "$enable_64bit" = "yes" -a $NBITS=32; then
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

      if test $sci_required_fortran = "yes"; then
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
      *gfortran*)
         FFLAGS="$FFLAGS $extra_cflags" 
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

# now that we know which NBITS the user wants, we can save this info
AC_SUBST(NBITS)
sci_defs_files="$sci_defs_files include/sci_defs/bits_testdefs.h"

###

AC_SUBST(WIN_DLLFLAGS)

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

##  --  Plume ----------------------

if test "$enable_plume" = "yes"; then
   BUILD_PLUME="yes"
else
   BUILD_PLUME="no"
fi
AC_SUBST(BUILD_PLUME)

## test for bison. yacc is not sufficient as it has no -o option.
## Only need for SCIRun2 currently.
AC_CHECK_PROG(YACC, bison, bison, nobison, $PATH /usr/bin)
if test "$enable_scirun2" = "yes" -a "$YACC" = "nobison"; then
  echo "configure: error: bison is required, and was not found in your PATH."
  echo "                  Please add it to your PATH, or set the environment"
  echo "                  variable YACC to the appropriate bison executable,"
  echo "                  then run configure again."
  exit
fi

AC_PROG_LEX
LEX_LIB_FLAG="$LEXLIB"
AC_SUBST(LEX_LIB_FLAG)

if test "$enable_scirun2" = "yes" -a "$LEX" != "flex" -a "$LEX" != "lex"; then
  echo "configure: error: flex or lex is required, and was not found in your PATH."
  echo "                  Please add it to your PATH, or set the environment"
  echo "                  variable LEX to the appropriate lex executable,"
  echo "                  then run configure again."
  exit
fi

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

##  Create the Packages/Uintah/tools dir for later use.
if test "$sci_required_tools" = "yes" ; then
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
sci_defs_files="$sci_defs_files include/sci_defs/template_testdefs.h"

##  set the default language after the build tools are found
AC_LANG([C])

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
  sci_tp_version=1.24.1
  sci_tp_arch=`uname -s`
  sci_compiler_vers=gcc-3.0.4

  # Try to figure out the compilers version (as it is used in the thirdparty directory name.)
  ac_try="$CC -v > /dev/null 2> conftest.out"
  (eval $ac_try) 2>&5
  vers=`grep "gcc version" conftest.out | sed -e 's/.*gcc version //' | cut -d' ' -f1`
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

  # extra thirdparty code for rtrt
  rtrt_tp_version=1.0.0
  rtrt_tp_path=$_sci_tp_base/rtrt/Thirdparty/$rtrt_tp_version/$_sci_tp_subdir

  # extra thirdparty code for plume
  plume_tp_version=1.0.0
  plume_tp_path=$_sci_tp_base/plume/Thirdparty/$plume_tp_version/$_sci_tp_subdir

  AC_MSG_RESULT(Using default thirdparty directory $sci_thirdparty_dir)
fi

if test ! -d "$sci_thirdparty_dir"; then
  AC_MSG_ERROR($sci_thirdparty_dir does not exist.)
fi

# This is where the lib actually lives:
if test ! -d "$sci_thirdparty_dir/lib"; then
  AC_MSG_ERROR(SCI Thirdparty invalid.  $sci_thirdparty_dir/lib does not exist.)
fi
if test ! -d "$sci_thirdparty_dir/include"; then
  AC_MSG_ERROR(SCI Thirdparty invalid.  $sci_thirdparty_dir/include does not exist.)
fi


sci_thirdparty_lib="$sci_thirdparty_dir/lib"
sci_thirdparty_include="$sci_thirdparty_dir/include"
SCI_THIRDPARTY_LIB_DIR="$sci_thirdparty_lib"
SCI_THIRDPARTY_LIB_DIR_FLAG="$LDRUN_PREFIX$sci_thirdparty_lib -L$sci_thirdparty_lib"
INC_SCI_THIRDPARTY_H=-I"$sci_thirdparty_include"
SCI_THIRDPARTY_DIR="$sci_thirdparty_dir"

AC_MSG_RESULT(Thirdparty library directory $SCI_THIRDPARTY_LIB_DIR)

SCI_SUBST_THIRDPARTY_DIR(INC_SCI_THIRDPARTY_H)
SCI_SUBST_THIRDPARTY_DIR(SCI_THIRDPARTY_LIB_DIR_FLAG)
AC_SUBST(SCI_THIRDPARTY_DIR)
AC_SUBST(INC_SCI_THIRDPARTY_H)
AC_SUBST(SCI_THIRDPARTY_LIB_DIR)
AC_SUBST(SCI_THIRDPARTY_LIB_DIR_FLAG)

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

SCI_TRY_LINK(M,math libraries and headers,math.h,,m,,,[
  double d = fabs(69.05);
],required)

AC_SUBST(M_LIB_FLAG)
AC_SUBST(M_LIB_DIR_FLAG)

##  --  search for X  ----------------------------------------------------

if test "$IS_WIN" = "yes"; then
  echo "skipping X test"
else
  _sci_optional=required
  if test "$with_X11_lib"; then
    _x_lib_dir="$with_X11_lib"
    _x_inc_dir="$with_X11_lib/../include"
    _sci_optional=specific
  else 
    _x_lib_dir="/usr/X11R6"
    BASE_LIB_PATH(_x_lib_dir,$_x_lib_dir)
    _x_inc_dir="/usr/X11R6/include"
  fi


  SCI_TRY_LINK(X,X libraries and headers,X11/Xlib.h,
               $_x_inc_dir,X11 Xext Xaw Xi Xt Xmu,$_x_lib_dir,,,$_sci_optional)

  if test "$X_LIB_DIR_FLAG" && test "$enable_rpath" = "no"; then
    AC_MSG_WARN("$X_LIB_DIR_FLAG")
    AC_MSG_WARN("flag for X has been changed to")
    X_LIB_DIR_FLAG="-L$LIB_DIR_X"
    AC_MSG_WARN("$X_LIB_DIR_FLAG")
  fi

fi # end "$IS_WIN" = "yes"

AC_SUBST(X_LIB_FLAG)
AC_SUBST(X_LIB_DIR_FLAG)

##  --  check for the required thread library  ---------------------------

SOCKET_LIBRARY=""

case $host in
  *-irix*)
    if test "$threads" = "unknown"; then
      threads=irix
    elif test "$threads" != "pthreads" -a "$threads" != "irix"; then
      SCI_MSG_ERROR(unknown thread type for host $host: $threads. exiting.)
    fi
    ;;
  *-pc-mingw*)
    threads=win32
    SOCKET_LIB_FLAG="-lws2_32"
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
   irix)
      SCI_TRY_LINK(FETCHOP,fetchop library,,,fetchop,,,,required)
      if test "$HAVE_FETCHOP" = "yes"; then
         CFLAGS="$CFLAGS -DSTL_SGI_THREADS"
         THREAD_LIB_FLAG="$FETCHOP_LIB_FLAG"
         DEF_THREAD=
      fi
      ;;
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
     THREAD_LIB_FLAG=-lwinmm
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

##  --  search for OpenGL  -----------------------------------------------

_sci_optional=required
if test "$with_opengl"; then
  _sci_lib="$with_opengl/lib"
  _sci_inc="$with_opengl/include"
  _sci_optional=specific
else
  BASE_LIB_PATH(_sci_lib,"/usr/X11R6")
  _sci_inc="/usr/X11R6/include"
fi

# GLU_OTHERS, GLU_OTHERS_FLAG

# These variables are used to supply libraries which GL and GLU may
# need.  As it turns out, the gl libraries in linux need the pthread
# library.  We need to make two versions of the variable though.
# GLU_OTHERS is used without the -l and GLU_OTHERS_FLAG includes it.
  
case $host in
  *linux*)
    if test "$threads" != "none"; then
      GLU_OTHERS="pthread m"
      GLU_OTHERS_FLAG="-lpthread -lm"
    else
      GLU_OTHERS="m"
      GLU_OTHERS_FLAG="-lm"
    fi
  ;;
  *)
  GLU_OTHERS=""
  GLU_OTHERS_FLAG=""
  ;;
esac

##########
##### This stuff was in the includes between gl.h and glu.hn
#####
##### #ifndef GLAPIENTRY
#####   #define GLAPIENTRY
##### #endif
#####
##### #ifndef GLAPI
#####   #define GLAPI
##### #endif

AC_LANG(C++)

if test "$IS_WIN" = "yes"; then
  SCI_TRY_LINK(GL,OpenGL libraries and headers,windows.h GL/gl.h GL/glu.h,
               $_sci_inc,glu32 opengl32 gdi32 $GLU_OTHERS,$_sci_lib,,,$_sci_optional)
else
  SCI_TRY_LINK(GL,OpenGL libraries and headers,GL/gl.h GL/glu.h GL/glx.h,
	       $_sci_inc,GLU GL $GLU_OTHERS,$_sci_lib,,,$_sci_optional)
fi

if test "$GL_LIB_DIR_FLAG" && test "$enable_rpath" = "no"; then
  AC_MSG_WARN("$GL_LIB_DIR_FLAG")
  AC_MSG_WARN("flag for OpenGL has been removed.")
  GL_LIB_DIR_FLAG=""
fi

AC_SUBST(GL_LIB_FLAG)
AC_SUBST(GL_LIB_DIR_FLAG)
AC_SUBST(INC_GL_H)

# Test for advanced volume rendering capability

SCI_COMPILE_LINK_TEST(EXT_HDRS,correct OpenGL extension headers ,
[
  #define GL_GLEXT_PROTOTYPES
  #include <GL/gl.h>
  #include <GL/glext.h>
],
$_sci_inc, GL, $_sci_lib,,
[
  int r = 5;
],optional)

if test "$HAVE_EXT_HDRS" = "yes"; then
  DEF_OGL_EXT_HDRS="#define CORRECT_OGLEXT_HDRS 1"
else
  DEF_OGL_EXT_HDRS=""
fi

if test "$avr" = "yes"; then
  AC_MSG_WARN([
  *** Including the Advance Volume Rendering (AVR) code in compilation.        ***
  *** Note: configure can erroneously determine this platform can use AVR.     ***
  *** use --disable-advanced-volume-rendering if compilation fails due to AVR. ***
  ])  
  COMPILE_AVR=1
  DEF_HAVE_AVR="#define HAVE_AVR_SUPPORT 1"
elif test "$avr" = "no"; then
  AC_MSG_WARN([
  *** Will not compile the advance volume rendering code. ***
  ])
  COMPILE_AVR=""
  DEF_HAVE_AVR=""
else
  DEF_HAVE_AVR=""
  
  if test "$HAVE_EXT_HDRS" = "yes"; then
    SCI_COMPILE_LINK_TEST(AVR, Advanced Volume OpenGL support,
    [
      #define GL_GLEXT_PROTOTYPES
      #include <GL/gl.h>
      #include <GL/glext.h>
    ],
    $_sci_inc, GL, $_sci_lib,,
    [
      glBindProgramARB(0,0);
      glActiveTexture(0);
    ],optional)
  else
    SCI_COMPILE_LINK_TEST(AVR, Advanced Volume OpenGL support,
    [
      #include <GL/gl.h>
    ],
    $_sci_inc, GL, $_sci_lib,,
    [
      glActiveTexture(0);
    ],optional)
  
  fi
      
  if test "$HAVE_AVR" = "yes"; then
  AC_MSG_WARN([
  *** Will include the advance volume rendering code in compilation.  ***
  *** configure can erroneously determine this platform can use avr.  ***
  *** use --disable-advanced-volume-rendering if compilation fails.   ***
  ])  
    COMPILE_AVR=1
    DEF_HAVE_AVR="#define HAVE_AVR_SUPPORT 1"
  fi
  
  AC_LANG(C)
  
fi
AC_SUBST(DEF_OGL_EXT_HDRS)
AC_SUBST(COMPILE_AVR)
AC_SUBST(DEF_HAVE_AVR)

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
SCI_TRY_LINK(STD_HDRS,standard headers,stdio.h stdlib.h stddef.h stdarg.h string.h limits.h errno.h fcntl.h float.h memory.h unistd.h ctype.h signal.h locale.h dirent.h assert.h sys/types.h sys/time.h sys/stat.h sys/timeb.h sys/param.h,,,,,,required)

# don't test for malloc.h or values.h on darwin
# don't test a bunch of then on win32
case $host in
  *-darwin*)
    SCI_TRY_LINK(STD_HDRS, standard headers-2,netdb.h sys/select.h sys/socket.h sys/mman.h sys/ioctl.h sys/resource.h sys/wait.h sys/utsname.h rpc/types.h netinet/in.h,,,,,,required)
    ;;
  *-irix*)
    # test for required sqi headers as well
    SCI_TRY_LINK(STD_HDRS,SGI standard headers,ieeefp.h ulocks.h values.h malloc.h netdb.h sys/select.h sys/socket.h sys/mman.h sys/ioctl.h sys/resource.h sys/wait.h sys/utsname.h rpc/types.h netinet/in.h,,,,,,required)
    DEF_IEEEFP="#define HAVE_IEEEFP_H 1"
    ;;
  *-pc-mingw*)
    ;;

  *)
    # default system headers that cannot be tested on all systems
    SCI_TRY_LINK(STD_HDRS,standard headers-2,values.h malloc.h netdb.h sys/select.h sys/socket.h sys/mman.h sys/ioctl.h sys/resource.h sys/wait.h sys/utsname.h rpc/types.h netinet/in.h,,,,,,required)
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

##  --  search for tcl stuff  ---------------------------------------------
echo 
AC_CHECKING(for required Tcl components...)
echo

_sci_optional=required
if test "$with_tcl"; then
  _sci_inc="$with_tcl/include"
  _sci_lib="$with_tcl/lib"
  _sci_optional=specific
else
  _sci_inc="$sci_thirdparty_include"
  _sci_lib="$sci_thirdparty_lib"
fi

if test -f "$_sci_lib/tclConfig.sh"; then 
  . $_sci_lib/tclConfig.sh; 
  if test ! -d $TCL_SRC_DIR; then
     # Catch the case of the SCIRun 3P being copied to a new location.
     AC_MSG_WARN(The TCL src dir specified in the $sci_lib/tclConfig.sh file)
     AC_MSG_WARN(is not valid.  This usually means that this SCIRun Thirdparty)
     AC_MSG_WARN(was copied to and not built in its current location.)
     AC_MSG_ERROR(This is not allowed.  Please rebuild the 3P or contact)
     AC_MSG_ERROR(scirun-develop@sci.utah.edu for more help.)
  fi
  _sci_tcl_inc="$_sci_inc $TCL_SRC_DIR/generic"
else
  _sci_tcl_inc="$_sci_inc"  
fi

SCI_TRY_LINK(TCL,tcl library and headers,tcl.h tclPort.h tclMath.h tclInt.h,
             $_sci_tcl_inc,tcl,$_sci_lib,,,$_sci_optional)

if test "$IS_WIN" = "yes"; then
  # change the unix-style dir to a windows-style dir for tclsh to read
  LIB_DIR_TCL=`echo "$LIB_DIR_TCL" | sed 's,/,,' | sed 's,/,:/,'`
fi

SCI_SUBST_THIRDPARTY_DIR(LIB_DIR_TCL)
SCI_SUBST_THIRDPARTY_DIR(TCL_LIB_DIR_FLAG)
SCI_SUBST_THIRDPARTY_DIR(INC_TCL_H)

AC_SUBST(LIB_DIR_TCL)
AC_SUBST(TCL_LIB_FLAG)
AC_SUBST(TCL_LIB_DIR_FLAG)
AC_SUBST(INC_TCL_H)

# -- tk --
if test -f "$_sci_lib/tkConfig.sh"; then 
  . $_sci_lib/tkConfig.sh; 
  _sci_tk_inc="$_sci_inc $TK_SRC_DIR/generic $TK_SRC_DIR/unix"
else
  _sci_tk_inc="$_sci_inc"  
fi

# Note: TK_DEFS (used below to help resolve tk issues) comes from tkConfig.sh
SCI_TRY_LINK(TK,tk library and headers, tk.h tk3d.h tkPort.h,
      $_x_inc_dir $_sci_tk_inc,tk $TCL_LIB_FLAG,$_sci_lib,$TK_DEFS $TCL_LIB_DIR_FLAG,,required)

SCI_SUBST_THIRDPARTY_DIR(TK_LIB_DIR_FLAG)
SCI_SUBST_THIRDPARTY_DIR(INC_TK_H)

AC_SUBST(TK_LIB_FLAG)
AC_SUBST(TK_LIB_DIR_FLAG)
AC_SUBST(INC_TK_H)

# -- itcl --
SCI_TRY_LINK(ITCL,itcl library and headers, itcl.h,
             $_sci_inc,itcl,$_sci_lib,$TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,required)

if test "$IS_WIN" = "yes"; then
  # change the unix-style dir to a windows-style dir for tclsh to read
  LIB_DIR_ITCL=`echo "$LIB_DIR_ITCL" | sed 's,/,,' | sed 's,/,:/,'`
fi

SCI_SUBST_THIRDPARTY_DIR(LIB_DIR_ITCL)
SCI_SUBST_THIRDPARTY_DIR(ITCL_LIB_DIR_FLAG_H)
SCI_SUBST_THIRDPARTY_DIR(INC_ITCL_H)

AC_SUBST(LIB_DIR_ITCL)
AC_SUBST(ITCL_LIB_FLAG)
AC_SUBST(ITCL_LIB_DIR_FLAG)
AC_SUBST(INC_ITCL_H)

# -- itk --
SCI_TRY_LINK(ITK,itk library and headers, itk.h, $_x_inc_dir $_sci_inc,itk,$_sci_lib,$ITCL_LIB_DIR_FLAG $ITCL_LIB_FLAG $TK_LIB_DIR_FLAG $TK_LIB_FLAG $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,required)

SCI_SUBST_THIRDPARTY_DIR(ITK_LIB_DIR_FLAG_H)
SCI_SUBST_THIRDPARTY_DIR(INC_ITK_H)

AC_SUBST(ITK_LIB_FLAG)
AC_SUBST(ITK_LIB_DIR_FLAG)
AC_SUBST(INC_ITK_H)

# -- blt --
SCI_TRY_LINK(BLT,BLT library and headers, blt.h,$_sci_inc,BLT,$_sci_lib,$TK_LIB_DIR_FLAG $TK_LIB_FLAG $TCL_LIB_DIR_FLAG $TCL_LIB_FLAG,,required) 

if test "$IS_WIN" = "yes"; then
  # Windows BLT needs these other libs
  BLT_LIB_FLAG="$BLT_LIB_FLAG -lgdi32 -lwinspool -lcomdlg32"
fi

SCI_SUBST_THIRDPARTY_DIR(BLT_LIB_DIR_FLAG)
SCI_SUBST_THIRDPARTY_DIR(INC_BLT_H)

AC_SUBST(BLT_LIB_FLAG)
AC_SUBST(BLT_LIB_DIR_FLAG)
AC_SUBST(INC_BLT_H)

# -- check TCL version == 8.3.2 and TK version == 8.3.2
CORRECT_TCL_VERSION=no
CORRECT_TK_VERSION=no
CORRECT_ITCL_VERSION=no

# if we don't have the tcl/tkConfig.sh files, we can grab the version number from the header file
if ! test -f "$_sci_lib/tclConfig.sh"; then 
  TCL_VERSION=`grep TCL_VERSION $sci_thirdparty_include/tcl.h | head -n 1 | awk '{print $3}' | sed 's,\",,g'`
  TCL_PATCH_LEVEL=".`grep TCL_RELEASE_SERIAL $sci_thirdparty_include/tcl.h | head -n 1 | awk '{print $3}'`"
  TCL_PREFIX=$sci_thirdparty_dir
fi

if ! test -f "$_sci_lib/tkConfig.sh"; then 
  TK_VERSION=`grep TK_VERSION $sci_thirdparty_include/tk.h | head -n 1 | awk '{print $3}' | sed 's,\",,g'`
  TK_PATCH_LEVEL=".`grep TK_RELEASE_SERIAL $sci_thirdparty_include/tk.h | head -n 1 | awk '{print $3}'`"
fi

SCI_CHECK_VAR_VERSION(Tcl,$TCL_VERSION$TCL_PATCH_LEVEL,8.3.2,,continue)
SCI_CHECK_VAR_VERSION(Tk,$TK_VERSION$TK_PATCH_LEVEL,8.3.2,,continue)
SCI_CHECK_VAR_VERSION(equal Tcl and Tk, $TCL_VERSION$TCL_PATCH_LEVEL,
                      $TK_VERSION$TK_PATCH_LEVEL,,continue,-ne)

AC_SUBST(TCL_PREFIX)
AC_SUBST(TCL_VERSION)
AC_SUBST(TK_VERSION)

AC_SUBST(ITCL_WIDGETS)

##  --  search for xerces-c++ --
_sci_optional=required
if test "$with_xercesc"; then
  _xer_lib_path="$with_xercesc/lib"
  _xer_inc="$with_xercesc/include"
  _sci_optional=specific
else
  _xer_lib_path="$sci_thirdparty_lib"
  _xer_inc="$sci_thirdparty_include"
fi

_xer_lib="xerces-c"
case $host in
  *aix*)
      # Needed on blue pacific for stricmp
      _xer_lib="xerces-c cfgdce"
      ;;
esac

SCI_TRY_LINK(XERCES_C,xerces library and headers, xercesc/dom/DOMNamedNodeMap.hpp xercesc/util/PlatformUtils.hpp xercesc/parsers/XercesDOMParser.hpp xercesc/dom/DOMNode.hpp xercesc/util/XMLUni.hpp xercesc/sax/SAXException.hpp xercesc/sax/SAXParseException.hpp xercesc/sax/ErrorHandler.hpp,$_xer_inc,$_xer_lib,$_xer_lib_path,,[
  DOMNode *node = 0;
  node->getNodeName();
],$_sci_optional)
  
SCI_SUBST_THIRDPARTY_DIR(XERCES_C_LIB_DIR_FLAG)
SCI_SUBST_THIRDPARTY_DIR(INC_XERCES_C_H)

AC_SUBST(XERCES_C_LIB_FLAG)
AC_SUBST(XERCES_C_LIB_DIR_FLAG)
AC_SUBST(INC_XERCESC_H)

##  ----------------------------------------------------------------------
##  ----------  required checks for optional libraries and headers  ------
##  ----------------------------------------------------------------------

##  --  search for optional headers  ------------------------------------

SCI_TRY_LINK(SSTREAM,sstream header,sstream,,,,,,optional)

if test "$HAVE_SSTREAM" = "yes"; then
   SSTREAM_COMPAT="no"
else
   # Use SCIRun sstream compatibility files...
   SSTREAM_COMPAT="yes"
fi

AC_SUBST(SSTREAM_COMPAT)

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
  SCI_TRY_LINK(EXC,WIN32 traceback library and headers,windows.h imagehlp.h,$_sci_inc,imagehlp,$_sci_lib,,,required)
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
       AC_MSG_CHECKING(that PNG is in requestion location '$with_png/lib')
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
    ./conftest-exc 2>&1 /dev/null

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
else
  _sci_lib="$sci_thirdparty_lib"
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

SCI_SUBST_THIRDPARTY_DIR(MPEGE_LIB_DIR_FLAG)

AC_SUBST(MPEGE_LIB_FLAG)
AC_SUBST(MPEGE_LIB_DIR_FLAG)
AC_SUBST(DEF_MPEG)

##  --  search for ImageMagick --------------------------------------

# Default values:
_sci_optional=optional
_sci_lib="$sci_thirdparty_lib"

if test "$with_magick" = "yes"; then
  # If --with-magick is specified without a parameter, then it makes it required
  _sci_optional=required
else 
  if test "$with_magick"; then
    # If --with-magick has a parameter, then use it
    _sci_lib="$with_magick/lib"
    BASE_LIB_PATH(_sci_lib,$_sci_lib)
    _sci_optional=specific
  fi 
fi

AC_MSG_CHECKING(for $_sci_optional ImageMagick)

if test ! -e $_sci_lib/../bin/Magick-config; then
   AC_MSG_WARN(Magick-config on in SCIRun Thirdparty installation.  Going to)
   AC_MSG_WARN(try to find it based on the location of the 'convert' program.)
   # ImageMagick not in SCIRun Thirdparty... Therefore it probably is
   # just in /usr/lib...
   MAGICK_DIR=`which convert`
   if test -z "$MAGICK_DIR"; then
      _sci_lib=$MAGICK_DIR/../lib
   else
      AC_MSG_WARN('convert' program not found...)
   fi
fi

found_magic_config="no"

if test -e $_sci_lib/../bin/Magick-config; then
    found_magic_config="yes"
    echo
    AC_MSG_RESULT(Using $_sci_lib/../bin/Magick-config to set ImageMagick variables.)

    MAG_LIBS=`$_sci_lib/../bin/Magick-config --libs`
    MAG_DIRS=`$_sci_lib/../bin/Magick-config --ldflags`

    #MAGICK_LIB_DIR_FLAG=
    #SCI_REMOVE_MINUS_L(MAGICK_LIB_FLAG,$MAG_DIRS,$MAG_LIBS)

    MAGICK_LIB_DIR_FLAG=`$_sci_lib/../bin/Magick-config --ldflags`
    MAGICK_LIB_FLAG=`$_sci_lib/../bin/Magick-config --libs`
    includes=`$_sci_lib/../bin/Magick-config --cppflags`

    SCI_TRY_LINK(MAGICK,ImageMagick library and headers,magick/magick.h,
                 $includes,$MAGICK_LIB_FLAG,$MAGICK_LIB_DIR_FLAG,,,required)

    DEF_MAGICK="#define HAVE_MAGICK 1"
    break
fi

if test "$found_magic_config" = "no"; then
   AC_MSG_RESULT(no.)
   if test "$_sci_optional" != "optional"; then
     echo
     AC_MSG_WARN(Cannot find a Magick-config file to get ImageMagick variables.)
     AC_MSG_WARN(This file usually resides in Magicklib_dir/../bin/Magick-config.)
     AC_MSG_WARN(Please install ImageMagick and then reconfigure.)
   fi
fi

if test "$with_magick" && test ! "$DEF_MAGICK"; then
  echo
  AC_MSG_ERROR(One or more of the ImageMagick components is missing.)
fi

SCI_SUBST_THIRDPARTY_DIR(MAGICK_LIB_DIR_FLAG)
SCI_SUBST_THIRDPARTY_DIR(INC_MAGICK_H)

AC_SUBST(MAGICK_LIB_FLAG)
AC_SUBST(MAGICK_LIB_DIR_FLAG)
AC_SUBST(INC_MAGICK_H)
AC_SUBST(DEF_MAGICK)

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
else
  _sci_lib="$sci_thirdparty_lib"
  _sci_inc="$sci_thirdparty_include"
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
elif test "$with_fortran"; then
  _sci_lib="$with_fortran/lib"
  _sci_inc="$with_fortran/include"
  _sci_optional=specific
fi

case $host in 
  *-irix*)
    SCI_TRY_LINK(F,fortran libraries,,,ftn fortran,$_sci_lib,,,$_sci_optional)
    ;;
  *aix*)
    # AIX Fortran Library Check
    SCI_TRY_LINK(F,fortran libraries,,,xlf90,$_sci_lib,,,$_sci_optional)
    ;;
  *linux*)
    # Linux Library Check
    if test "$F77" = "pgf77"; then
      SCI_TRY_LINK(F,fortran libraries,,,pgftnrtl,$_sci_lib,,,$_sci_optional)
    elif test "$F77" = "gfortran-4.0"; then
      SCI_TRY_LINK(F,fortran libraries,,,gfortran,$_sci_lib,,,$_sci_optional)
    elif test "$F77" = "ifort"; then
      SCI_TRY_LINK(F,fortran libraries,,,ifcore,$_sci_lib,,,$_sci_optional)
    else
      SCI_TRY_LINK(F,fortran libraries,,,g2c,$_sci_lib,,,$_sci_optional)
    fi
    ;;
  *dec*)
    SCI_TRY_LINK(F,fortran libraries,,,for Ufor,$_sci_lib,,,$_sci_optional)
    ;;
  *-darwin*)
    # OSX Library Check
    if test "$F77" = "xlf"; then
      SCI_TRY_LINK(F,fortran libraries,,$_sci_inc,xlf90,$_sci_lib,,,$_sci_optional)
    else
      SCI_TRY_LINK(F,fotran libraries,,$_sci_inc,gfortran,$_sci_lib,,,optional)
      if test "$HAVE_F" = "no"; then
            SCI_TRY_LINK(F,fortran libraries,,$_sci_inc,g2c,$_sci_lib,,,$_sci_optional)
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

AC_SUBST(F_LIB_FLAG)
AC_SUBST(F_LIB_DIR_FLAG)

##  --  search for Camal (Sandia meshing library) ------------------------

_sci_optional=optional

if test "$with_camal" = "yes"; then
  # If --with-camal is specified without a parameter, then it makes it required
  _sci_optional=required
else 
  if test "$with_camal"; then
    # If --with-camal has a parameter, then use it
    _sci_lib="$with_camal"
    _sci_inc="$with_camal/include"
    BASE_LIB_PATH(_sci_lib,$_sci_lib)
    _sci_optional=specific
  fi 
fi

if test "$_sci_optional" != "optional" && test "$HAVE_F" != "yes"; then
   AC_MSG_ERROR(Fortran library not found but is required for CAMAL!)
fi

if test "$IS_OSX" = "yes"; then
  SCI_TRY_LINK(CAMAL,CAMAL library,,$_sci_inc,CMLTet210 CMLUtl210,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,,$_sci_optional)
  if test "$HAVE_CAMAL" = "yes"; then
    CAMAL_LIB_DIR_FLAG="$CAMAL_LIB_DIR_FLAG -Wl,-read_only_relocs,warning -single_module"
  fi
else
  SCI_TRY_LINK(CAMAL,CAMAL library,,$_sci_inc,CMLTet202,$_sci_lib,$F_LIB_DIR_FLAG $F_LIB_FLAG,,$_sci_optional)
fi

if test "$HAVE_CAMAL" = "yes"; then
  DEF_CAMAL="#define HAVE_CAMAL 1"
fi

if test "$with_camal" && test ! "$DEF_CAMAL"; then
  echo
  AC_MSG_ERROR(One or more of the CAMAL components is missing.)
  echo
fi

AC_SUBST(CAMAL_LIB_FLAG)
AC_SUBST(INC_CAMAL_H)
AC_SUBST(CAMAL_LIB_DIR_FLAG)
AC_SUBST(DEF_CAMAL)
AC_SUBST(HAVE_CAMAL)

##  --  search for atlas stuff  -------------------------------------------

if test "$with_blas" && test "$with_atlas"; then
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
                  -faltivec,,,-framework vecLib,,$_sci_optional)
     ;;
   *)
     SCI_TRY_LINK(BLAS,atlas version of libraries and headers,cblas.h,
             $_sci_inc,cblas f77blas atlas,$_sci_lib,$F_LIB_FLAG,,$_sci_optional)
     ;;
esac

HAVE_CBLAS=$HAVE_BLAS
DEF_CBLAS=$DEF_BLAS

if test "$HAVE_BLAS" = "yes"; then
  DEF_BLAS="#define HAVE_BLAS 1"
  DEF_CBLAS="#define HAVE_CBLAS 1"
  sci_required_blas=no
fi


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
                      essl,$_sci_lib,$F_LIB_FLAG,,required)
       ;;
    *dec*)
       SCI_TRY_LINK(BLAS,cxml library,,,
                      cxml,$_sci_lib,$F_LIB_FLAG,,required)
       ;;
    *)
       # Can't use the "specific" key word as we are testing for a number
       # of different configurations.
       SCI_TRY_LINK(BLAS,blas and cblas libraries-1,cblas.h,
                    $_sci_inc,cblas blas,$_sci_lib,$F_LIB_FLAG,,optional)
       if test "$HAVE_BLAS" = "yes"; then
          HAVE_CBLAS=yes
	  AC_SUBST(HAVE_CBLAS)
	  DEF_CBLAS="#define HAVE_CBLAS 1"
	  AC_SUBST(DEF_CBLAS)
       else
          SCI_TRY_LINK(BLAS,blas and scs libraries-2,cblas.h,
            $_sci_inc,blas scs_mp,$_sci_lib,$F_LIB_FLAG,,optional)
          if test "$HAVE_BLAS" = "no"; then
             #scs library should be used on the SGI altix
             #SCI_TRY_LINK(BLAS,blas and scs libraries-2,cblas.h,
             #   $_sci_inc,blas scs_mp,$_sci_lib,$F_LIB_FLAG,,optional)
             SCI_TRY_LINK(BLAS,blas and cblas libraries-2,cblas.h,
               $_sci_inc,blas,$_sci_lib,$F_LIB_FLAG,,optional)
             if test "$HAVE_BLAS" = "yes"; then
                HAVE_CBLAS=yes
                AC_SUBST(HAVE_CBLAS)
                DEF_CBLAS="#define HAVE_CBLAS 1"
                AC_SUBST(DEF_CBLAS)
             else
                SCI_TRY_LINK(BLAS,blas library,,,blas,$_sci_lib,$F_LIB_FLAG,,optional)
                if test "$HAVE_BLAS" = "no"; then
	           SCI_TRY_LINK(BLAS,optional fblas library,,,
		   	     fblas,$_sci_lib,$F_LIB_FLAG,,optional)
                else
                   # Verify it was in the right spot
                   if test "$_sci_specific" = "yes"; then
                     SCI_TRY_LINK(BLAS,blas library,,,blas,$_sci_lib,$F_LIB_FLAG,,specific)
                   fi
                fi
             fi
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

# DAV, FIX ME
# _sci_optional=optional
#if test "$sci_required_lapack" = "yes" ; then
#   _sci_optional=required
#fi

if test "$HAVE_F" = "yes" && test "$HAVE_BLAS" = "yes"; then
  if test "$with_lapack"; then
    _sci_lib="$with_lapack"
  else
    _sci_lib="$LIB_DIR_BLAS"
  fi
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

if test "$with_lapack" && test ! "$DEF_LAPACK"; then
  echo
  AC_MSG_ERROR(One or more of the LAPACK components is missing.)
  echo
fi

AC_SUBST(HAVE_LAPACK)
AC_SUBST(DEF_LAPACK)
AC_SUBST(LAPACK_LIB_FLAG)
AC_SUBST(LAPACK_LIB_DIR_FLAG)

AC_SUBST(HAVE_LAPACKMP)
AC_SUBST(LAPACKMP_LIB_FLAG)
AC_SUBST(LAPACKMP_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/lapack_testdefs.h"

##  --  search for mpi  --------------------------------------------------

if test "$sci_required_mpi" = "yes"; then
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
        _sci_optional=optional
        if test "$with_mpi"; then
           if test "$with_mpi" = "yes"; then
              _sci_optional=required
           else
              _sci_optional=specific
              _sci_lib="$with_mpi/lib"
              _sci_inc="$with_mpi/include"
           fi
        fi

	# After each attempt to locate mpi, we need to verify that the mpi it
	# chose is in the path we specified.  This is because some machines
	# have MPI installed in a standard location and sometimes will choose
	# that one over the one we specified.
	the_file=libmpi
	
        # /usr/include/mpi is the standard location of mpi on some linux dists.
        SCI_TRY_LINK(MPI,mpi library and header-1,mpi.h,
                     $_sci_inc /usr/include/mpi,mpi lam util pmpi,$_sci_lib,,,optional)

        if test "$HAVE_MPI" = "yes" && test "$_sci_optional" = "specific" &&
           test ! -e $with_mpi/lib/$the_file.so &&
           test ! -e $with_mpi/lib/$the_file.a &&
           test ! -e $with_mpi/lib/$the_file.dylib; then
             AC_MSG_WARN(Found libmpi but not in requested directory: $with_mpi/lib)
	     HAVE_MPI="no"
        fi

        if test "$HAVE_MPI" != "yes"; then
          # Try without pmpi (and with pthreads) (for the newer version
          # of lam mpi.
          # Use -DMPIPP_H so we don't include mpi++.h where we don't want it.

          SCI_TRY_LINK(MPI,mpi library and header-2,mpi.h,
          $_sci_inc /usr/include/mpi,mpi lam util pthread,$_sci_lib,-DMPIPP_H,,optional)

          # verify that it was found in the location requested.
          if test "$HAVE_MPI" = "yes" && test "$_sci_optional" = "specific" &&
             test ! -e $with_mpi/lib/$the_file.so &&
             test ! -e $with_mpi/lib/$the_file.a &&
             test ! -e $with_mpi/lib/$the_file.dylib; then
             AC_MSG_WARN(Found libmpi but not in requested directory: $with_mpi/lib)
	     HAVE_MPI="no"
          fi

          if test "$HAVE_MPI" != "yes"; then
             # If building with newer LAM on OSX, pthreads and util not needed.
             SCI_TRY_LINK(MPI,mpi library and header-3,mpi.h,
             $_sci_inc /usr/include/mpi,lam mpi,$_sci_lib,-DMPIPP_H,,optional)

             # verify that it was found in the location requested
             if test "$HAVE_MPI" = "yes" && test "$_sci_optional" = "specific" &&
                test ! -e $with_mpi/lib/$the_file.so &&
                test ! -e $with_mpi/lib/$the_file.a &&
                test ! -e $with_mpi/lib/$the_file.dylib; then
                AC_MSG_WARN(Found libmpi but not in the requested directory: $with_mpi/lib)
                HAVE_MPI="no"
             fi                        

             if test "$HAVE_MPI" != "yes"; then
                SCI_TRY_LINK(MPI,mpi library and header-4,mpi.h,
                $_sci_inc /usr/include/mpi,mpi,$_sci_lib,-DMPIPP_H,,optional)

                # verify that it was found in the location requested.
                if test "$HAVE_MPI" = "yes" && test "$_sci_optional" = "specific" &&
                   test ! -e $with_mpi/lib/$the_file.so &&
                   test ! -e $with_mpi/lib/$the_file.a &&
                   test ! -e $with_mpi/lib/$the_file.dylib; then
                   AC_MSG_WARN(Found libmpi but not in requested directory: $with_mpi/lib)
                   HAVE_MPI="no"
                fi

                if test "$HAVE_MPI" != "yes"; then
                   SCI_TRY_LINK(MPI,mpich library and header,mpi.h,
                   $_sci_inc /usr/include/mpi,mpich pmpich,$_sci_lib,,,optional)

                   if test "$HAVE_MPI" = "yes"; then
                      HAVE_MPICH="yes"
                   fi
                fi
             fi
          fi
        fi
        
        if test "$HAVE_MPI" = "yes"; then

           if test "$_sci_optional" = "specific"; then
              the_file=libmpi
              if test "$HAVE_MPICH" = "yes"; then
                the_file=libmpich
              fi

              # verify that it was found in the location requested.
              if test ! -e $with_mpi/lib/$the_file.so &&
                 test ! -e $with_mpi/lib/$the_file.a &&
                 test ! -e $with_mpi/lib/$the_file.dylib; then
                   echo
                   AC_MSG_ERROR(Found libmpi but not in requested directory: $with_mpi/lib)
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
fi #end sci_required_mpi

AC_SUBST(HAVE_MPI)
AC_SUBST(DEF_MPI)
AC_SUBST(INC_MPI_H)
AC_SUBST(MPI_LIB_FLAG)
AC_SUBST(MPI_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/mpi_testdefs.h"

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
PETSC_VERSION_DIR=petsc-2.2.1

case $host in
  *-irix*)
    if test "$enable_64bit" = "yes"; then
      PETSC_ARCH=IRIX64
    else
      PETSC_ARCH=IRIX32
    fi
    ;;
  *linux*)
    if test "$enable_64bit" = "yes"; then
      PETSC_ARCH=linux64
    else
      PETSC_ARCH=linux
    fi
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
  *-pc-mingw*)
    PETSC_ARCH=win
    ;;
  *)
    PETSC_ARCH=""
    ;;
esac

# check for mpi-uni if either --with-unipetsc or --with-mpiuni,
#   check here to look in petsc's subdirs.
#  --  search for mpi-uni  --------------------------------------------
# Again required for the following uni-petsc checks, that attempt to support
# the MatrixSolver module.

_sci_lib=
_sci_inc=

if test "$with_mpiuni" = "yes"; then
  echo
  AC_MSG_ERROR(You must specify a directory with the --with-mpiuni=dir.)
  echo
elif test "$with_unipetsc"; then
  _sci_lib="$with_unipetsc/lib/$PETSC_SUB/$PETSC_ARCH"
  if test ! -d "$_sci_lib" || test ! -d "$with_unipetsc/include/mpiuni" ||
     test ! -d "$with_unipetsc/bmake/$PETSC_ARCH" ; then
    echo
    AC_MSG_WARN(Specified unipetsc directory is invalid.  One of the following dirs does not exist:)
    AC_MSG_WARN($_sci_lib)
    AC_MSG_WARN($with_unipetsc/include/mpiuni)
    AC_MSG_WARN($with_unipetsc/bmake/$PETSC_ARCH)
    AC_MSG_ERROR(Please fix this problem and reconfigure.)
  fi
  _sci_inc="$with_unipetsc/include/mpiuni $with_unipetsc/bmake/$PETSC_ARCH" 
fi

SCI_TRY_LINK(MPI_UNI,mpi-uni libraries and headers,mpi.h,$_sci_inc,mpiuni,$_sci_lib,,,optional)

if test "$HAVE_MPI_UNI" != "yes"; then
   SCI_TRY_LINK(MPI_UNICH,mpiunich and pmpiunich libraries,,
                  $_sci_inc,mpiunich pmpiunich,$_sci_lib,,,optional)

   if test "$HAVE_MPI_UNICH" = "yes"; then
     LIB_DIR_MPI_UNI="$LIB_DIR_MPI_UNICH"
     MPI_UNI_LIB_FLAG="$MPI_UNICH_LIB_FLAG"
     MPI_UNI_LIB_DIR_FLAG="$MPI_UNICH_LIB_DIR_FLAG"
   fi
fi

if test "$HAVE_MPI_UNI" = "yes"; then
   if test "$HAVE_MPI_UNICH" = "yes"; then
     DEF_MPI_UNI="#define HAVE_MPI_UNICH 1"
   else
     DEF_MPI_UNI="#define HAVE_MPI_UNI 1"
   fi
fi

if test "$with_mpiuni" && test ! "$DEF_MPI_UNI"; then
  echo
  AC_MSG_ERROR(One or more of the MPI_UNI components is missing.)
  echo
fi

if test "$with_unipetsc" && test ! "$DEF_MPI_UNI"; then
  echo
  AC_MSG_ERROR(One or more of the MPI_UNI components is missing.)
  echo
fi

AC_SUBST(DEF_MPI_UNI)
AC_SUBST(INC_MPI_UNI_H)
AC_SUBST(MPI_UNI_LIB_FLAG)
AC_SUBST(MPI_UNI_LIB_DIR_FLAG)


#  --  search for PETSc  --------------------------------------------
if test -z "$PETSC_ARCH"; then
   AC_MSG_ERROR(PETSC_ARCH environment var is not set.  Please set.)
fi

# Make sure we have either mpi or mpi-uni, blas, and lapack
# It seems that under AIX and OSF, the libs that cover blas also cover lapack
if (test "$HAVE_MPI" = "yes" || test "$HAVE_MPI_UNI" = "yes") && test "$HAVE_BLAS" = "yes" && 
   (test "$PETSC_ARCH" = "rs6000_sp" || test "$PETSC_ARCH" = "dec" ||
    test "$HAVE_LAPACK" = "yes"); then

  _sci_lib_dir=
  _sci_inc=

  if test "$with_petsc"; then
    _sci_lib_dir="$with_petsc/lib/$PETSC_SUB/$PETSC_ARCH"
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
  
      _sci_lib_dir="$_sci_lib_dir $i/$PETSC_VERSION_DIR/lib/$PETSC_SUB/$PETSC_ARCH"
      _sci_inc="$_sci_inc $i/$PETSC_VERSION_DIR/include $i/$PETSC_VERSION_DIR/bmake/$PETSC_ARCH"
    done  
  fi

# For Petsc 2.2 use this line below:   petsc petscvec petscmat petscdm petscsnes petscts petscksp $LAPACK_LIB_FLAG $BLAS_LIB_FLAG $M_LIB_FLAG $MPI_LIB_FLAG $F_LIB_FLAG $DL_LIB_FLAG,
# Use -DMPIPP_H so we don't include mpi++.h where we don't want it.

  SCI_TRY_LINK(PETSC,
               petsc libraries and headers,
               petsc.h,
               $_sci_inc /usr/include/petsc,
               petsc petscvec petscmat petscdm petscsnes petscts petscksp $LAPACK_LIB_FLAG $BLAS_LIB_FLAG $M_LIB_FLAG $MPI_LIB_FLAG $F_LIB_FLAG $DL_LIB_FLAG,
               $_sci_lib_dir $LAPACK_LIB_DIR_FLAG $BLAS_LIB_DIR_FLAG $MPI_LIB_DIR_FLAG,
               $INC_MPI_H -DMPIPP_H,
               ,
               optional)

  if test "$HAVE_PETSC" = "yes" ; then
    DEF_PETSC="#define HAVE_PETSC 1"
  else
    # Give it a shot on the old petsc version if we're not doing Uintah
    if test "$pkg_Uintah" != "yes"; then
      SCI_TRY_LINK(PETSC,
                   petsc libraries and headers (old),
                   petsc.h,
                   $_sci_inc /usr/include/petsc,
                   petsc petscvec petscmat petscdm petsccontrib petscsnes petscts petscsles $LAPACK_LIB_FLAG $BLAS_LIB_FLAG $M_LIB_FLAG $MPI_LIB_FLAG $F_LIB_FLAG $DL_LIB_FLAG,
                   $_sci_lib_dir $LAPACK_LIB_DIR_FLAG $BLAS_LIB_DIR_FLAG $MPI_LIB_DIR_FLAG,
                   $INC_MPI_H -DMPIPP_H,
                   ,
                   optional)
      if test "$HAVE_PETSC" = "yes"; then
        DEF_PETSC="#define HAVE_PESC 1"
      fi
    fi  

  fi    
    
  if test "$with_petsc" && test ! "$DEF_PETSC"; then
    echo
    AC_MSG_ERROR(One or more of the PETSc components is missing.)
    echo
  fi

  #  --  search for uni-petsc  --------------------------------------------

  if test "$HAVE_PETSC" != "yes" ; then

    # Only check for uni petsc if petsc not found.
    _sci_lib=""
    _sci_inc=""

    if test "$with_unipetsc"; then
      _sci_lib="$with_unipetsc/lib/$PETSC_SUB/$PETSC_ARCH"
      _sci_inc="$with_unipetsc/bmake/$PETSC_ARCH $with_unipetsc/include"
    fi

# For petsc 2.2 use this line below:  petsc petscvec petscmat petscdm petscsnes petscts petscksp,
# Use -DMPIPP_H so we don't include mpi++.h where we don't want it.
    SCI_TRY_LINK(PETSC_UNI,
                 petsc-uni libraries and headers,
                 petsc.h,
                 $_sci_inc /usr/include/petsc,
                 petsc petscvec petscmat petscdm petscsnes petscts petscksp,
                 $_sci_lib,
                 $LAPACK_LIB_DIR_FLAG $LAPACK_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $M_LIB_FLAG $INC_MPI_UNI_H $MPI_UNI_LIB_DIR_FLAG $MPI_UNI_LIB_FLAG $F_LIB_FLAG -DMPIPP_H,
                 ,
                 optional)

    if test "$HAVE_PETSC_UNI" = "yes" ; then
      DEF_PETSC_UNI="#define PETSC_UNI 1"
    else
    # Give it a shot on the old petsc-uni version
      SCI_TRY_LINK(PETSC_UNI,
                   petsc libraries and headers (old),
                   petsc.h,
                   $_sci_inc /usr/include/petsc,
                   petsc petscvec petscmat petscdm petscsnes petscts,
                   $_sci_lib,
                   $LAPACK_LIB_DIR_FLAG $LAPACK_LIB_FLAG $BLAS_LIB_DIR_FLAG $BLAS_LIB_FLAG $M_LIB_FLAG $INC_MPI_UNI_H $MPI_UNI_LIB_DIR_FLAG $MPI_UNI_LIB_FLAG $F_LIB_FLAG -DMPIPP_H,
                   ,
                   optional)
      if test "$HAVE_PETSC_UNI" = "yes" ; then
        DEF_PETSC_UNI="#define PETSC_UNI 1"
      fi  
    fi


   
    if test "$with_unipetsc" && test ! "$DEF_PETSC_UNI"; then
      echo
      AC_MSG_ERROR(One or more of the UNI-PETSC components is missing.)
      echo
    fi
  fi

else
  AC_MSG_WARN(One or more of MPI or BLAS or LAPACK is missing.)
  AC_MSG_WARN(HAVE_MPI: $HAVE_MPI)
  AC_MSG_WARN(HAVE_MPI_UNI: $HAVE_MPI_UNI)
  AC_MSG_WARN(HAVE_BLAS: $HAVE_BLAS)
  AC_MSG_WARN(HAVE_LAPACK: $HAVE_LAPACK)
  AC_MSG_WARN(Not configuring PETSc.)
fi

AC_SUBST(HAVE_PETSC)
AC_SUBST(PETSC_LIB_FLAG)
AC_SUBST(PETSC_LIB_DIR_FLAG)
AC_SUBST(DEF_PETSC)
AC_SUBST(INC_PETSC_H)

AC_SUBST(HAVE_PETSC_UNI)
AC_SUBST(PETSC_UNI_LIB_FLAG)
AC_SUBST(PETSC_UNI_LIB_DIR_FLAG)
AC_SUBST(DEF_PETSC_UNI)
AC_SUBST(INC_PETSC_UNI_H)
sci_defs_files="$sci_defs_files include/sci_defs/petsc_testdefs.h"

##  --  search for Insight  ---------------------------------------------------

if test "$sci_required_insight" = "yes"; then

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
    INC_INSIGHT_H="`cat $dir/ITKConfig.cmake | grep 'SET(ITK_INCLUDE_DIRS' | sed 's/SET(ITK_INCLUDE_DIRS \"/;/' | sed 's/\")//' | sed 's/;/ -I/g' | tr -d '\r'`"
    insight_libs_dir="`cat $dir/ITKConfig.cmake | grep 'SET(ITK_LIBRARY_DIRS' | sed 's/SET(ITK_LIBRARY_DIRS \"/;/' | sed 's/\")//' | sed 's/;//g' | tr -d '\r'`"
    INSIGHT_LIB_DIR_FLAG="$LDRUN_PREFIX$insight_libs_dir -L$insight_libs_dir"
    INSIGHT_LIB_FLAG="-lITKFEM -lITKIO -lITKMetaIO -lITKAlgorithms -lITKStatistics -lITKBasicFilters -lITKCommon  -litkvnl -litkvnl_algo -litknetlib -litksys -lITKDICOMParser"

    if test -a "$insight_libs_dir/libitkzlib.$SO_OR_ARCHIVE_EXTENSION"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litkzlib"
    elif test "$HAVE_Z" = "yes"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG $Z_LIB_DIR_FLAG $Z_LIB_FLAG"
    fi

    if test -a "$insight_libs_dir/libitktiff.$SO_OR_ARCHIVE_EXTENSION"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litktiff"
    elif test "$HAVE_TIFF" = "yes"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG $TIFF_LIB_DIR_FLAG $TIFF_LIB_FLAG"
    fi


    INSIGHT_JPEG_LINKED=no

    if test -a "$insight_libs_dir/libitkjpeg8.$SO_OR_ARCHIVE_EXTENSION"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litkjpeg8"
       INSIGHT_JPEG_LINKED=yes
    elif test "$HAVE_JPEG" = "yes" && test "$INSIGHT_JPEG_LINKED" = "no" ; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG $JPEG_LIB_DIR_FLAG $JPEG_LIB_FLAG"
       INSIGHT_JPEG_LINKED=yes
    fi

    if test -a "$insight_libs_dir/libitkjpeg12.$SO_OR_ARCHIVE_EXTENSION"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litkjpeg12"
       INSIGHT_JPEG_LINKED=yes
    elif test "$HAVE_JPEG" = "yes" && test "$INSIGHT_JPEG_LINKED" = "no" ; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG $JPEG_LIB_DIR_FLAG $JPEG_LIB_FLAG"
       INSIGHT_JPEG_LINKED=yes
    fi

    if test -a "$insight_libs_dir/libitkjpeg16.$SO_OR_ARCHIVE_EXTENSION"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litkjpeg16"
       INSIGHT_JPEG_LINKED=yes
    elif test "$HAVE_JPEG" = "yes" && test "$INSIGHT_JPEG_LINKED" = "no" ; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG $JPEG_LIB_DIR_FLAG $JPEG_LIB_FLAG"
       INSIGHT_JPEG_LINKED=yes
    fi

    if test -a "$insight_libs_dir/libitkjpeg.$SO_OR_ARCHIVE_EXTENSION" && test "$INSIGHT_JPEG_LINKED" = "no"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litkjpeg"
       INSIGHT_JPEG_LINKED=yes
    elif test "$HAVE_JPEG" = "yes" && test "$INSIGHT_JPEG_LINKED" = "no"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG $JPEG_LIB_DIR_FLAG $JPEG_LIB_FLAG"
       INSIGHT_JPEG_LINKED=yes
    fi

    if test -a "$insight_libs_dir/libITKNrrdIO.$SO_OR_ARCHIVE_EXTENSION"; then
       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -lITKNrrdIO"
    fi

    if ! test "$HAVE_PNG" = "yes"; then
      echo "note: using libitkpng"
      INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litkpng"
    else
      INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG $PNG_LIB_DIR_FLAG $PNG_LIB_FLAG"
    fi


#    INSIGHT_VERSION_MINOR="`cat $dir/ITKConfig.cmake | grep 'SET(ITK_VERSION_MINOR' | sed 's/SET(ITK_VERSION_MINOR \"//' | sed 's/\")//' | tr -d '\r'`"
#    INSIGHT_VERSION_MAJOR="`cat $dir/ITKConfig.cmake | grep 'SET(ITK_VERSION_MAJOR' | sed 's/SET(ITK_VERSION_MAJOR \"//' | sed 's/\")//' | tr -d '\r'`"
#    if test "$INSIGHT_VERSION_MAJOR" = "2"; then
#       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -lITKNrrdIO -litktiff -litkjpeg8 -litkjpeg12 -litkjpeg16"
#    elif test "$INSIGHT_VERSION_MINOR" = "9"; then
#       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litktiff -litkjpeg8 -litkjpeg12 -litkjpeg16"
#    elif test "$INSIGHT_VERSION_MINOR" = "8" || test "$INSIGHT_VERSION_MINOR" = "6"; then
#       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litktiff -litkjpeg"
#    elif test "$INSIGHT_VERSION_MINOR" = "4"; then
#       INSIGHT_LIB_FLAG="$INSIGHT_LIB_FLAG -litkjpeg"
#    fi


#    INSIGHT_LIB_FLAG="`cat $dir/ITKConfig.cmake | grep 'SET(ITK_LIBRARIES' | sed 's/SET(ITK_LIBRARIES//' | sed 's/)//' | sed 's/ / -l/g'`" 

    DEF_INSIGHT="#define HAVE_INSIGHT"
    HAVE_INSIGHT="yes"

    AC_MSG_RESULT($dir)
  else
    AC_MSG_RESULT(not found.)
  fi
 
  if test "$with_insight" && test ! "$DEF_INSIGHT"; then
    echo
    AC_MSG_ERROR(One or more of the INSIGHT components is missing.)
    echo
  fi

fi

AC_SUBST(HAVE_INSIGHT)
AC_SUBST(DEF_INSIGHT)
AC_SUBST(INC_INSIGHT_H)
AC_SUBST(INSIGHT_LIB_FLAG)
AC_SUBST(INSIGHT_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/insight_testdefs.h"

##  --  search for Ptolemy II ---------------------------------------------------
if test "$sci_required_ptolemyII" = "yes"; then

  AC_MSG_CHECKING(for Java.)

  # check for Java
  # Java version check should go here
  if test -z "$with_java"; then

    AC_MSG_WARN(Java is required by the Ptolemy package.)
    AC_MSG_CHECKING(for environment variable JAVA_HOME or JAVAHOME)

    if test -z "$JAVA_HOME"; then
      if test -z "$JAVAHOME"; then # Ptolemy build uses JAVAHOME

	AC_MSG_WARN(Package Ptolemy requires Java.)
	AC_MSG_ERROR(Use the --with-java=DIR option or set the environment variable JAVA_HOME to the directory that contains your Java distribution.)

      else # JAVAHOME set in environment
	jdir="$JAVAHOME"
      fi
    else # JAVA_HOME set in environment
      jdir="$JAVA_HOME"
    fi
  else # --with-java
    jdir="$with_java"
  fi

  if test -d "$jdir"; then
    AC_MSG_RESULT($jdir)

    JAVA_PATH="$jdir"
  else

    AC_MSG_ERROR(Java installation directory is missing.)

  fi

  AC_MSG_CHECKING(for Ptolemy II.)


  # search here
  if test "$with_ptolemyII"; then
    pdir="$with_ptolemyII"
  fi

  if test -d "$pdir"; then
    AC_MSG_RESULT($pdir)

    PTOLEMY_PATH="$pdir"
    PTOLEMY_CLASSPATH="$PTOLEMY_PATH:/lib/diva.jar:/lib/java_cup.jar:/lib/sootclasses.jar:/lib/jasminclasses.jar:/lib/ptCal.jar:/lib/ptjacl.jar:/lib/jython.jar:/lib/mapss.jar:/lib/saxon7.jar:/ptolemy/ptolemy.jar:/ptolemy/ptsupport.jar:/ptolemy/vergil/vergil.jar:$JAVA_PATH/jre:/lib/rt.jar"
    HAVE_PTOLEMY="yes"
    DEF_PTOLEMY="#define HAVE_PTOLEMY 1"
  else
    AC_MSG_ERROR(Ptolemy II directory is missing.)
  fi

fi

AC_SUBST(HAVE_PTOLEMY)
AC_SUBST(DEF_PTOLEMY)
AC_SUBST(JAVA_PATH)
AC_SUBST(PTOLEMY_PATH)
AC_SUBST(PTOLEMY_CLASSPATH)
sci_defs_files="$sci_defs_files include/sci_defs/ptolemy_testdefs.h"

##  --  search for xalan  ---------------------------------------------------
if test "$sci_required_xalan" ; then
  AC_MSG_CHECKING(for xalan-j.)

  if test "$with_xalan"; then
	dir="$with_xalan"
  else
	dir="$sci_thirdparty_dir"
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

##  --  search for Babel  ----------------------------------------------------

_sci_lib=
_sci_inc=
_sci_optional=optional

if test "$with_babel" = "yes"; then
  _sci_optional=required
elif test "$with_babel"; then
  _sci_lib="$with_babel/lib"
  _sci_inc="$with_babel/include"
  _sci_optional=specific
fi

SCI_TRY_LINK(SIDL,sidl library and headers,sidl.h,$_sci_inc,sidl,$_sci_lib,,,$_sci_optional)
  
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
fi 

AC_SUBST(HAVE_BABEL)
AC_SUBST(DEF_BABEL)
AC_SUBST(BABEL_BIN)

AC_SUBST(INC_SIDL_H)
AC_SUBST(SIDL_LIB_FLAG)
AC_SUBST(SIDL_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/babel_testdefs.h"

##  --  search for netsolve ----------------------------------------------

_sci_lib=
_sci_inc=
_sci_optional=optional

if test "$with_netsolve" = "yes"; then
  _sci_optional=required
elif test "$with_netsolve"; then
  _sci_lib="$with_netsolve/lib"
  _sci_inc="$with_netsolve/include"
  _sci_optional=specific
fi

SCI_TRY_LINK(NETSOLVE,netsolve library,,,netsolve,$_sci_lib,,,$_sci_optional)

AC_SUBST(NETSOLVE_LIB_FLAG)
AC_SUBST(NETSOLVE_LIB_DIR_FLAG)


##  --  search for glut  -----------------------------------------------

if test "$sci_required_glut" = "yes"; then

   libs="$GL_LIB_DIR_FLAG $GL_LIB_FLAG $X_LIB_FLAG $GLU_OTHERS_FLAG"

   if test "$with_glut"; then
     _sci_lib_glut="$with_glut"
     BASE_LIB_PATH(_sci_lib_glut,$_sci_lib_glut)
     if test "$with_opengl"; then
       _sci_lib_ogl="$with_opengl"
       BASE_LIB_PATH(_sci_lib_ogl,$_sci_lib_ogl)
       _sci_lib="$_sci_lib_glut $_sci_lib_ogl"
       _sci_inc="$with_glut/include $with_opengl/include"
     else
       _sci_lib_X="/usr/X11R6"
       BASE_LIB_PATH(_sci_lib_X,$_sci_lib_X)
       _sci_lib="$_sci_lib_glut $_sci_lib_X"
       _sci_inc="$with_glut/include /usr/X11R6/include"
     fi
   else
     if test "$with_opengl"; then
       _sci_lib="$sci_thirdparty_lib $with_opengl/lib"
       _sci_inc="$sci_thirdparty_include $with_opengl/include"
     else
       _sci_lib="$sci_thirdparty_lib /usr/X11R6/lib"
       _sci_inc="$sci_thirdparty_include /usr/X11R6/include"
     fi
   fi

   SCI_TRY_LINK(GLUT,glut library and headers,GL/glut.h,$_sci_inc,glut,$_sci_lib,$libs,,required)

  if test "$GLUT_LIB_DIR_FLAG" && test "$enable_rpath" = "no"; then
    AC_MSG_WARN("$GLUT_LIB_DIR_FLAG")
    AC_MSG_WARN("flag for GLUT has been removed.")
    GLUT_LIB_DIR_FLAG=""
  fi

fi #end sci_required_glut

AC_SUBST(INC_GLUT_H)
AC_SUBST(GLUT_LIB_FLAG)
AC_SUBST(GLUT_LIB_DIR_FLAG)


##  --  search for glui  -----------------------------------------------
if test "$sci_required_glui" = "yes"; then
   if test "$with_glui"; then
   
     ## Set the path for glut and glui
     _sci_lib_glut="$with_glut"
     BASE_LIB_PATH(_sci_lib_glut,$_sci_lib_glut)
     _sci_lib_glui="$with_glui"
     BASE_LIB_PATH(_sci_lib_glui,$_sci_lib_glui)
     
     if test "$with_opengl"; then
       ## OpenGL path
       _sci_lib_ogl="$with_opengl"
       BASE_LIB_PATH(_sci_lib_ogl,$_sci_lib_ogl)
       _sci_lib="$_sci_lib_glut $_sci_lib_glui $_sci_lib_ogl"
       _sci_inc="$with_glut/include $with_glui/include $with_opengl/include"
     else
       ## X path
       _sci_lib_X="/usr/X11R6"
       BASE_LIB_PATH(_sci_lib_X,$_sci_lib_X)
       _sci_lib="$_sci_lib_glut $_sci_lib_glui $_sci_lib_X"
       _sci_inc="$with_glut/include $with_glui/include /usr/X11R6/include"
     fi
   else
     if test "$with_opengl"; then
       _sci_lib="$sci_thirdparty_lib $with_opengl/lib"
       _sci_inc="$sci_thirdparty_include $with_opengl/include"
     else
       _sci_lib="$sci_thirdparty_lib /usr/X11R6/lib"
       _sci_inc="$sci_thirdparty_include /usr/X11R6/include"
     fi
   fi

   SCI_TRY_LINK(GLUI,glui library and headers,GL/glut.h glui.h,
                $_sci_inc,glui,$_sci_lib,$libs $GLUT_LIB_DIR_FLAG $GLUT_LIB_FLAG,,required)

  if test "$GLUI_LIB_DIR_FLAG" && test "$enable_rpath" = "no"; then
    AC_MSG_WARN("$GLUI_LIB_DIR_FLAG")
    AC_MSG_WARN("flag for GLUI has been removed.")
    GLUI_LIB_DIR_FLAG=""
  fi

fi # end sci_required_glui

AC_SUBST(INC_GLUI_H)
AC_SUBST(GLUI_LIB_FLAG)
AC_SUBST(GLUI_LIB_DIR_FLAG)


##  --  search for oogl  -----------------------------------------------
if test "$sci_required_oogl" = "yes"; then
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
    SCI_TRY_LINK(OOGL,oogl library and headers,oogl/basicTexture.h oogl/shadedPrim.h,
                 $_sci_inc,oogl,$_sci_lib,$INC_GLUT_H $INC_GLUI_H,,optional)
  fi

  if test "$HAVE_OOGL" = "yes"; then
    DEF_OOGL="#define HAVE_OOGL 1"
  fi

  if test "$with_oogl" && test ! "$DEF_OOGL"; then
    echo
    AC_MSG_ERROR(One or more of the OOGL components is missing.)
    echo
  fi

  if test "$OOGL_LIB_DIR_FLAG" && test "$enable_rpath" = "no"; then
    AC_MSG_WARN("$OOGL_LIB_DIR_FLAG")
    AC_MSG_WARN("flag for OOGL has been removed.")
    OOGL_LIB_DIR_FLAG=""
  fi

fi # end sci_required_oogl

AC_SUBST(DEF_OOGL)
AC_SUBST(INC_OOGL_H)
AC_SUBST(OOGL_LIB_FLAG)
AC_SUBST(OOGL_LIB_DIR_FLAG)

##  --  search for glew  -----------------------------------------------
if test "$sci_required_glew" = "yes"; then
  if test "$with_glew"; then
    # first find the opengl headers and libraries
    if test "$with_opengl"; then
      _sci_lib="$with_glew/lib $with_opengl/lib"
      _sci_inc="$with_glew/include $with_opengl/include"
    else
      _sci_lib="$with_glew/lib /usr/X11R6/lib"
      _sci_inc="$with_glew/include /usr/X11R6/include"
    fi
  else
    if test "$with_opengl"; then
      _sci_lib="/usr/X11R6/lib $with_opengl/lib"
      _sci_inc="/usr/X11R6/include $with_opengl/include"
    else
      _sci_lib="/usr/X11R6/lib"
      _sci_inc="/usr/X11R6/include"
    fi        
  fi

# Was:
#    #include<GL/glew.h>
#
#    #ifndef GLAPIENTRY
#      #define GLAPIENTRY
#    #endif
#
#    #ifndef GLEWAPI
#      #define GLEWAPI
#    #endif
#    #include<GL/glxew.h>

  SCI_TRY_LINK(GLEW,glew libraries and headers,GL/glew.h GL/glxew.h,
               $_sci_inc, GLEW,$_sci_lib,,,optional)

  if test "$HAVE_GLEW" = "yes"; then
    DEF_GLEW="#define HAVE_GLEW 1"
  fi  

  if test "$with_glew" && test ! "$DEF_GLEW"; then
    echo
    AC_MSG_ERROR(One or more of the GLEW components is missing.)
    echo
  fi

  if test "$GLEW_LIB_DIR_FLAG" && test "$enable_rpath" = "no"; then
    AC_MSG_WARN("$GLEW_LIB_DIR_FLAG")
    AC_MSG_WARN("flag for GLEW has been removed.")
    GLEW_LIB_DIR_FLAG=""
  fi

fi # end sci_required_glew

AC_SUBST(DEF_GLEW)
AC_SUBST(INC_GLEW_H)
AC_SUBST(GLEW_LIB_FLAG)
AC_SUBST(GLEW_LIB_DIR_FLAG)

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
    SCI_TRY_LINK(FREETYPE,freetype library and headers,ft2build.h,
                 $_sci_inc,freetype,$_sci_lib,$includes,,$_sci_optional)

    INC_FREETYPE_H="$INC_FREETYPE_H $includes"

    DEF_FREETYPE="#define HAVE_FREETYPE 1"
fi

if test "$found_freetype_config" = "no"; then
   AC_MSG_RESULT(No 'freetype-config' file found in $freetype_bin.  Skipping Freetype check.)
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


##  --  search for ftgl stuff  -------------------------------------------
_sci_lib=
_sci_inc=
_sci_optional=optional

if test "$with_ftgl" = "yes"; then
  _sci_optional=required
elif test "$with_ftgl"; then
  _sci_lib="$with_ftgl/lib"
  _sci_inc="$with_ftgl/include"
  _sci_optional=specific
fi

SCI_TRY_LINK(FTGL,ftgl library,FTGL/FTGLTextureFont.h,$_sci_inc,ftgl,$_sci_lib,$INC_FREETYPE_H $FREETYPE_LIB_DIR_FLAG $FREETYPE_LIB_FLAG,,$_sci_optional)

if test "$HAVE_FTGL" = "yes"; then
  DEF_FTGL="#define HAVE_FTGL 1"
fi

if test "$with_ftgl" && test ! "$DEF_FTGL"; then
  echo
  AC_MSG_ERROR(One or more of the FTGL components is missing.)
fi

AC_SUBST(FTGL_LIB_FLAG)
AC_SUBST(FTGL_LIB_DIR_FLAG)
AC_SUBST(DEF_FTGL)
AC_SUBST(INC_FTGL_H)

sci_defs_files="$sci_defs_files include/sci_defs/ogl_testdefs.h"


##  --  search for audio  -----------------------------------------------

if test "$sci_required_audio" = "yes"; then
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

fi # end sci_required_audio

AC_SUBST(DEF_AUDIO)
AC_SUBST(HAVE_AUDIO)
AC_SUBST(AUDIO_LIB_FLAG)
AC_SUBST(AUDIO_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/audio_testdefs.h"

##  --  search for teem  -------------------------------------------------

# Default values: (Really required, but need to give info to user if fails.)
#                 (See below.)
_sci_optional=optional
_sci_lib="$sci_thirdparty_lib"
_sci_inc="$sci_thirdparty_include"

if test "$with_teem"; then
  if test "$with_teem" = "yes"; then
    AC_MSG_WARN(Teem is already required - you do not need to specify --with-teem.)
  else
    # If --with-teem has a parameter, then use it
    _sci_lib="$with_teem/lib"
    _sci_inc="$with_teem/include"
    BASE_LIB_PATH(_sci_lib,$_sci_lib)
    _sci_optional=specific
  fi 
fi

# Determine if teem is using png.  Assume teem is either in the 3rd Party
# or where the user has specified (as we are going to use nm to look at its
# symbols and are not trying to link against it (at this point.))

result=`nm $_sci_lib/libteem.a | grep png_write_info`
if test _sci_optional != "optional" && test -n "$result" && test "$HAVE_PNG" != "yes"; then
   AC_MSG_WARN(Teem was built with PNG support.  However the SCIRun configure)
   AC_MSG_WARN(did not find PNG.  You need to determine where it is... as)
   AC_MSG_WARN(Teem seems to have found it already.  You will most likely then need)
   AC_MSG_WARN(to use --with-png=/dir/above/dir/having/libpng on your configure line.)
   AC_MSG_ERROR(Please fix this problem and then configure again.)
fi

SCI_TRY_LINK(TEEM,teem library and headers,teem/nrrd.h teem/ten.h,$_sci_inc,teem $PNG_LIB_FLAG $Z_LIB_FLAG -lm,$_sci_lib $PNG_LIB_DIR_FLAG $Z_LIB_DIR_FLAG,,[
float a[3], b[9], c[7];
tenEigensolve_f(a,b,c);
]
,$_sci_optional)

if test "$HAVE_TEEM" = "yes"; then
   DEF_TEEM="#define HAVE_TEEM 1"
else
   AC_MSG_WARN(Teem thirdparty library check failed.  This most likely)
   AC_MSG_WARN(means one of two things.  1. Could not find the teem)
   AC_MSG_WARN(thirdparty library files.  or 2. Your teem thirdparty)
   AC_MSG_WARN(is too old.  Look in config.log for more information.)
   AC_MSG_WARN(If the error message is something like:)
   AC_MSG_WARN(`tenEigensolve_f' undeclared then you most likely)
   AC_MSG_WARN(have too old of a thirdparty.)
   AC_MSG_ERROR(One or more of the TEEM components is missing.)   
fi

SCI_SUBST_THIRDPARTY_DIR(TEEM_LIB_DIR_FLAG)
SCI_SUBST_THIRDPARTY_DIR(INC_TEEM_H)

AC_SUBST(HAVE_TEEM)
AC_SUBST(DEF_TEEM)
AC_SUBST(INC_TEEM_H)
AC_SUBST(TEEM_LIB_FLAG)
AC_SUBST(TEEM_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/teem_testdefs.h"

##  --  search for mdsplus  -------------------------------------------------

if test "$sci_required_mdsplus" = "yes"; then
  if test "$with_mdsplus"; then
    _sci_lib="$with_mdsplus/lib"
    _sci_inc="$with_mdsplus/include"
  else
    _sci_lib="/usr/local/mdsplus/lib"
    _sci_inc="/usr/local/mdsplus/include"
  fi

# mdsplus is a C lib so it has to be checked as a C file not CC.
  cxx="$CXX"
  CXX="$CC"

  ext="$ac_ext"
  ac_ext="c"

  SCI_TRY_LINK(MDSPLUS,mdslib library and headers,mdslib.h,$_sci_inc,MdsLib_client,$_sci_lib,,,optional)

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
fi #end sci_required_mdsplus

AC_SUBST(DEF_MDSPLUS)
AC_SUBST(INC_MDSPLUS_H)
AC_SUBST(MDSPLUS_LIB_FLAG)
AC_SUBST(MDSPLUS_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/mdsplus_testdefs.h"  

##  --  search for hdf5  -------------------------------------------------

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

  SCI_TRY_LINK(HDF5,hdf5 library and headers,hdf5.h,$_sci_inc,hdf5,$_sci_lib,,,$_sci_optional)

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

##  --  search for perfex  -----------------------------------------------

if test "$sci_required_perfex" = "yes"; then

  _sci_lib=

  SCI_TRY_LINK(PERFEX,perfex library,,,perfex,$_sci_lib,,,required)

  if test "$HAVE_PERFEX" = "yes"; then
    DEF_PERFEX="#define HAVE_PERFEX 1"
  fi
fi

AC_SUBST(DEF_PERFEX)
AC_SUBST(PERFEX_LIB_FLAG)
AC_SUBST(PERFEX_LIB_DIR_FLAG)


##  --  search for crypto --------------------------------

if test "$sci_required_crypto" = "yes"; then
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
fi # end sci_required_crypto

AC_SUBST(DEF_CRYPTO)
AC_SUBST(CRYPTO_LIB_FLAG)
AC_SUBST(CRYPTO_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/crypto_testdefs.h"

##  --  search for ssl ---------------------------------

if test "$sci_required_ssl" = "yes"; then

  _sci_lib=

  if test "$with_ssl"; then
    _sci_lib="$with_ssl/lib"
  else
    _sci_lib="$sci_thirdparty_lib"
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
fi # end sci_required_ssl

AC_SUBST(DEF_SSL)
AC_SUBST(SSL_LIB_FLAG)
AC_SUBST(SSL_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/ssl_testdefs.h"

##  --  globus nexus  ----------------------------------------------------

if test "$sci_required_globus" = "yes"; then

  if test "$with_globus"; then
    _sci_lib="$with_globus/lib"
    _sci_inc="$with_globus/include"
  else
    _sci_lib="$sci_thirdparty_lib"
    _sci_inc="$sci_thirdparty_include"
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
  if test "$with_globus"; then
    with_globus_lib=$with_globus
  else
    with_globus_lib=$LIB_DIR_GLOBUS/..
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

fi #end sci_required_globus

AC_SUBST(HAVE_GLOBUS)
AC_SUBST(DEF_GLOBUS)
AC_SUBST(INC_GLOBUS_H)
AC_SUBST(GLOBUS_LIB_FLAG)
AC_SUBST(GLOBUS_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/globus_testdefs.h"

##  --  search for Qt  ----------------------------------------------------

_sci_lib=
_sci_inc=
_sci_optional=optional

if test "$with_qt" = "yes"; then
  _sci_optional=required
elif test "$with_qt"; then
  _sci_lib="$with_qt/lib"
  _sci_inc="$with_qt/include"
  _sci_optional=specific
fi

qt_bin=$_sci_lib/../bin
# Only looking for one .h file... assuming all the rest are in the same place.
SCI_TRY_LINK(QT_MT,qt library,qcanvas.h,$_sci_inc,qt-mt Xxf86vm,$_sci_lib /usr/X11R6/lib,,,optional)   
QT_MOC_BIN=""
AS_MESSAGE(checking for moc in $qt_bin...)
AC_CHECK_PROG(QT_MOC_BIN,moc,$qt_bin/moc,NOMOC,)
  
if test "$HAVE_QT_MT" = "yes" && test $QT_MOC_BIN != "NOMOC"; then
   DEF_QT="#define HAVE_QT 1"
   HAVE_QT="yes"
fi
  
if test "$with_qt" && test ! "$DEF_QT"; then
  echo
  AC_MSG_ERROR(One or more of the QT components is missing.)
fi

AC_SUBST(HAVE_QT)
AC_SUBST(DEF_QT)
AC_SUBST(INC_QT_MT_H)
AC_SUBST(QT_MT_LIB_FLAG)
AC_SUBST(QT_MT_LIB_DIR_FLAG)
AC_SUBST(QT_MOC_BIN)
sci_defs_files="$sci_defs_files include/sci_defs/qt_testdefs.h"

##  --  search for ruby  ----------------------------------------------------
if test "$sci_required_ruby" = "yes"; then
  _sci_optional=optional

  if test "$with_ruby" = "yes"; then
    # --with-ruby has been specified without a parameter.
    _sci_lib="$sci_thirdparty_lib"
    _ruby_inc="$sci_thirdparty_include"
    _sci_optional=required
  else 
   if test "$with_ruby"; then
     #if --with-ruby has been specified with a parameter, then use that parameter
    _sci_lib="$with_ruby/lib"
    # A standard ruby install only places include files in platform-specific directories and does
    # not bother to create symlinks.  We have to search for the .h files here.  This may fail where
    # multiple builds of ruby are placed under the same root install tree.  If "find" fails, then the
    # configure will also fail unless ruby.h is in a standard path.
    echo "Searching for ruby.h under $with_ruby"
    _ruby_inc=`find $with_ruby -name "ruby.h" -print | sed 's/ruby.h//'`
    _sci_optional=specific
   fi
  fi
  
  # Only looking for one .h file... assuming all the rest are in the same place.
  SCI_TRY_LINK(RUBY,ruby library,ruby.h,$_ruby_inc,ruby-static,$_sci_lib,,,$_sci_optional)

  if test "$HAVE_RUBY" = "yes"; then
     DEF_RUBY="#define HAVE_RUBY 1"
     HAVE_RUBY="yes"
  fi
  
  if test "$with_ruby" && test ! "$DEF_RUBY"; then
    echo
    AC_MSG_ERROR(One or more of the ruby components are missing.)
    echo
  fi
fi # end sci_required_ruby

AC_SUBST(HAVE_RUBY)
AC_SUBST(DEF_RUBY)
AC_SUBST(INC_RUBY_H)
AC_SUBST(RUBY_LIB_FLAG)
AC_SUBST(RUBY_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/ruby_testdefs.h"

##  --  search for tao corba ----------------------------------------------------
if test "$sci_required_tao" = "yes"; then
  if test "$with_tao" = "yes"; then
    # --with-tao has been specified without a parameter.
    _sci_lib="$sci_thirdparty_lib"
    _tao_inc="$sci_thirdparty_include"
  else
   if test "$with_tao"; then
    _sci_lib="$with_tao/ace"
    _tao_inc="$with_tao $with_tao/ace $with_tao/TAO $with_tao/TAO/tao"
   fi
  fi
                                                                                                                                                
  # Only looking for one .h file... assuming all the rest are in the same place.
  SCI_TRY_LINK(TAO,tao corba library,pre.h,$_tao_inc,TAO_IORTable TAO_PortableServer TAO_IORInterceptor TAO ACE,$_sci_lib,,,optional)
                                                                                                                                                
  if test "$HAVE_TAO" = "yes"; then
     DEF_TAO="#define HAVE_TAO 1"
     HAVE_TAO="yes"
  fi
                                                                                                                                                
  if test "$with_tao" && test ! "$DEF_TAO"; then
    echo
    AC_MSG_ERROR(One or more of the Tao components are missing.)
    echo
  fi
fi # end sci_required_tao
                                                                                                                                                
AC_SUBST(HAVE_TAO)
AC_SUBST(DEF_TAO)
AC_SUBST(INC_TAO_H)
AC_SUBST(TAO_LIB_FLAG)
AC_SUBST(TAO_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/tao_testdefs.h"

##  --  search for vtk  ----------------------------------------------------
if test "$sci_required_vtk" = "yes"; then
  if test "$with_vtk"; then
    _sci_lib="$with_vtk/lib/vtk"
    _vtk_inc="$with_vtk/include/vtk"
  else
    _sci_lib="$sci_thirdparty_lib"
    _vtk_inc="$sci_thirdparty_include"
  fi
  
  # Only looking for one .h file... assuming all the rest are in the same place.
  SCI_TRY_LINK(VTK,vtk library, vtkImageData.h,$_vtk_inc,vtkCommon vtkFiltering vtkGraphics vtkImaging vtkRendering vtkIO vtkjpeg vtkpng vtkzlib,$_sci_lib,,,optional)


  if test "$HAVE_VTK" = "yes"; then
     DEF_VTK="#define HAVE_VTK 1"
     HAVE_VTK="yes"
  fi
  
  if test "$with_vtk" && test ! "$DEF_VTK"; then
    echo
    AC_MSG_ERROR(One or more of the vtk components are missing.)
    echo
  fi
fi # end sci_required_vtk

AC_SUBST(HAVE_VTK)
AC_SUBST(DEF_VTK)
AC_SUBST(INC_VTK_H)
AC_SUBST(VTK_LIB_FLAG)
AC_SUBST(VTK_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/vtk_testdefs.h"

##  --  search for VDT  ----------------------------------------------------

if test "$sci_required_vdt" = "yes"; then
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

    SCI_TRY_LINK(VDT,vdt libraries and headers,vdt.h,$_vdt_inc,$_vdt_libs,$_vdt_lib_path,$MISC_LIB_FLAGS,[
    VDT_triangulate();
    ],required) 
 
    if test "$HAVE_VDT" = "yes"; then
      DEF_VDT="#define HAVE_VDT 1"
      break
    fi
  done

  if test "$with_vdt" && test ! "$DEF_VDT"; then
    echo
    AC_MSG_ERROR(One or more of the VDT components is missing.)
    echo
  fi
fi # end sci_required_vdt

AC_SUBST(DEF_VDT)
AC_SUBST(INC_VDT_H)
AC_SUBST(VDT_LIB_FLAG)
AC_SUBST(VDT_LIB_DIR_FLAG)
sci_defs_files="$sci_defs_files include/sci_defs/vdt_testdefs.h"

##  --  search for hypre stuff  ------------------------------------------

_sci_lib=
_sci_inc=
_sci_optional=optional

if test "$with_hypre" = "yes"; then
  _sci_optional=required
elif test "$with_hypre"; then
  _sci_lib="$with_hypre/lib"
  _sci_inc="$with_hypre/include"
  _sci_optional=specific
fi

# We used to also check for HYPRE_blas, but hypre 1.9 doesn't have it,
# so we aren't going to check for it in any version.  If the other 3
# libs exist, we will assume that HYPRE_blas is there if it is needed.
SCI_TRY_LINK(HYPRE,HYPRE libraries and headers,HYPRE.h,$_sci_inc,
                         HYPRE_struct_mv HYPRE_struct_ls HYPRE_LSI,
                         $_sci_lib,,,$_sci_optional)

if test "$HAVE_HYPRE" = "yes" ; then
  DEF_HYPRE="#define HAVE_HYPRE 1"

  if test -f $LIB_DIR_HYPRE/../include/HYPRE_config.h; then
    # Determine the version of hypre.
    found_it=`grep "PACKAGE_VERSION" $LIB_DIR_HYPRE/../include/HYPRE_config.h | grep 1.9`
    DEF_HYPRE_1_9="#define HAVE_HYPRE_1_9 0"
    HAVE_HYPRE_1_9="no"
    if test -n "$found_it"; then
       DEF_HYPRE_1_9="#define HAVE_HYPRE_1_9 1"
       HAVE_HYPRE_1_9="yes"
    fi
  fi

fi

if test "$with_hypre" && test ! "$DEF_HYPRE"; then
  echo
  AC_MSG_ERROR(One or more of the HYPRE components is missing.)
fi

AC_SUBST(HAVE_HYPRE)
AC_SUBST(HAVE_HYPRE_1_9)
AC_SUBST(DEF_HYPRE_1_9)
AC_SUBST(DEF_HYPRE)
AC_SUBST(INC_HYPRE_H)
AC_SUBST(HYPRE_LIB_FLAG)
AC_SUBST(HYPRE_LIB_DIR_FLAG)

sci_defs_files="$sci_defs_files include/sci_defs/hypre_testdefs.h"

##  --  search for boost stuff  -------------------------------------------
if test "$sci_required_boost" = "yes"; then
  _sci_optional=required

  if test "$with_boost" = ""; then
    _sci_lib=/usr/local/lib/
    _sci_inc="/usr/local/include"
  else 
  if test "$with_boost" = "yes"; then
    # --with-boost has been specified without a parameter.
    _sci_lib=/usr/local/lib/
    _sci_inc=/usr/local/include
  else 
   if test "$with_boost"; then
     #if --with-boost has been specified with a parameter, then use that parameter
    _sci_lib="$with_boost/lib"
    _sci_inc="$with_boost/include"
    _sci_optional=specific
   fi
  fi
  fi
  
  
  case $host in
    *linux*)
       boost_ver=gcc-mt
       ;;
    *-darwin*)
       boost_ver=1_32
       ;;
    *)
       #assume there is no specific version
       ;;
  esac

  # Only looking for one .h file... assuming all the rest are in the same place.
  SCI_TRY_LINK(BOOST,boost library,boost/program_options.hpp,$_sci_inc,boost_program_options-$boost_ver,$_sci_lib,,,$_sci_optional)

  if test "$HAVE_BOOST" = "yes"; then
     DEF_BOOST="#define HAVE_BOOST"
     HAVE_BOOST="yes"
  fi
  
  if test "$with_boost" && test ! "$DEF_BOOST"; then
    echo
    AC_MSG_ERROR(One or more of the boost components are missing.)
    echo
  fi
fi # end sci_required_boost


AC_SUBST(BOOST_LIB_FLAG)
AC_SUBST(BOOST_LIB_DIR_FLAG)
AC_SUBST(DEF_BOOST)
AC_SUBST(INC_BOOST_H)

sci_defs_files="$sci_defs_files include/sci_defs/boost_testdefs.h"

##  --  search for loki stuff  -------------------------------------------
if test "$sci_required_loki" = "yes"; then
  _sci_optional=required

  if test "$with_loki" = ""; then
    _sci_lib=/usr/local/lib/
    _sci_inc="/usr/local/include"
  else 
  if test "$with_boost" = "yes"; then
    # --with-loki has been specified without a parameter.
    _sci_lib=/usr/local/lib/
    _sci_inc=/usr/local/include
  else 
   if test "$with_boost"; then
     #if --with-loki has been specified with a parameter, then use that parameter
    _sci_lib="$with_loki"
    _sci_inc="$with_loki"
    _sci_optional=specific
   fi
  fi
  fi
  
  # Only looking for one .h file... assuming all the rest are in the same place.
   SCI_TRY_LINK(LOKI,loki library,loki/Singleton.h,$_sci_inc,loki,$_sci_lib,,,$_sci_optional)

  if test "$HAVE_LOKI" = "yes"; then
    DEF_LOKI="#define HAVE_LOKI"
    HAVE_LOKI="yes"
  fi

  if test "$with_loki" && test ! "$DEF_LOKI"; then
    echo
    AC_MSG_ERROR(One or more of the loki components are missing.)
    echo
  fi
fi # end sci_required_loki

AC_SUBST(LOKI_LIB_FLAG)
AC_SUBST(LOKI_LIB_DIR_FLAG)
AC_SUBST(DEF_LOKI)
AC_SUBST(INC_LOKI_H)

sci_defs_files="$sci_defs_files include/sci_defs/loki_testdefs.h"


##  --  search for TENA  ----------------------------------------------------

_sci_lib=
_sci_inc=
_sci_optional=optional

if test "$with_tena" = "yes"; then
  _sci_optional=required
elif test "$with_tena"; then
  _sci_lib="$with_tena/lib"
  _sci_inc="$with_tena/5.1/include"
  TENA_HOME="$with_tena"
  _sci_optional=specific
fi

# fixme
HAVE_TENA="yes" 
INC_TENA_H="-I$_sci_inc"
TENA_LIB_DIR="-L$_sci_lib/"
TENA_LIB_DIR_FLAG="-Wl,-rpath -Wl,$TENA_LIB_DIR"
TENA_BIN_DIR="$with_tena/5.1/bin"

#SCI_TRY_LINK(TENA,sidl library and headers,sidl.h,$_sci_inc,sidl,$_sci_lib,,,$_sci_optional)
  
AC_CHECKING(for TENA in $TENA_HOME)
#AC_CHECK_PROG(BABEL_BIN,babel,$LIB_DIR_SIDL/../bin/babel,NONE,$LIB_DIR_SIDL/../bin)


if test "$HAVE_TENA" = "yes" ; then
   DEF_TENA="#define HAVE_TENA 1"
   HAVE_TENA="yes"
fi

TENA_VERSION="5.1"
TENA_PLATFORM="fc3-gcc343"
TENA_OMC_VERSION="0"

PS_DEFINES="-D__EXTENSIONS__ -D_PTHREADS -D_POSIX_C_SOURCE=199506L -D_POSIX_PTHREAD_SEMANTICS -DTENA_PLATFORM=$TENA_PLATFORM"

TENA_DEFINES="-D_REENTRANT $PS_DEFINES"

AC_SUBST(HAVE_TENA)
AC_SUBST(DEF_TENA)
AC_SUBST(TENA_BIN_DIR)

AC_SUBST(INC_TENA_H)
AC_SUBST(TENA_LIB_DIR)
AC_SUBST(TENA_LIB_DIR_FLAG)
AC_SUBST(TENA_VERSION)
AC_SUBST(TENA_PLATFORM)
AC_SUBST(TENA_OMC_VERSION)
AC_SUBST(TENA_DEFINES)

sci_defs_files="$sci_defs_files include/sci_defs/tena_testdefs.h"


##  --  search for TENA_OM  ----------------------------------------------------

if test "$with_tena_om" = "yes"; then
  _tena_om=$TENA_HOME
  HAVE_TENA_OM="yes" 
  _sci_optional=required
elif test "$with_tena_om"; then
  _tena_om="$with_tena_om"
  HAVE_TENA_OM="yes" 
  _sci_optional=specific
fi



if test "$HAVE_TENA_OM" = "yes" ; then
   DEF_TENA_OM="#define HAVE_TENA_OM 1"
   HAVE_TENA_OM="yes"

   # fixme
   TENA_OM_DEF_DIR="$_tena_om/$TENA_VERSION/include/OMs"
   TENA_OM_IMPL_DIR="$_tena_om/$TENA_VERSION/src"

fi
  
if test "$with_tena" && test ! "$DEF_TENA_OM"; then
  echo
  AC_MSG_ERROR(One or more of the TENA_OM components is missing.)
fi 

AC_SUBST(HAVE_TENA_OM)
AC_SUBST(DEF_TENA_OM)

AC_SUBST(TENA_OM_DEF_DIR)
AC_SUBST(TENA_OM_IMPL_DIR)
#sci_defs_files="$sci_defs_files include/sci_defs/tena_defs.h"


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

if test "$sci_required_uuid" = "yes"; then
  AC_CHECK_HEADERS(sys/uuid.h uuid/uuid.h)
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
      if test "$UUID_LIB_FLAG"; then
        AC_MSG_RESULT($UUID_LIB_FLAG)
      else
        AC_MSG_RESULT(none required)
      fi
      HAVE_UUID="yes"
    else
      AC_MSG_RESULT(not found)
      have_uuid=no
      HAVE_UUID="no"
    fi
    LIBS="$old_LIBS"
  fi
fi # end sci_required_uuid

AC_SUBST(HAVE_UUID)
AC_SUBST(UUID_LIB_FLAG)

##  --  See if we have everything needed for SCIRun2 ----------------------

if test "$enable_scirun2" = "yes"; then
  if test "$have_uuid" != "yes"; then
    case $host in
      *linux*)
        AC_MSG_WARN([uuid_generate function and libuuid not found.  Will use custom uuid generation instead.])
        ;;
      ## For now, don't check for uuid if on macintosh.
      *-darwin*)
        AC_MSG_WARN([Skipping uuid check on MAC. Will use custom uuid generation instead.])
        ;;
      *)
        AC_MSG_WARN([cannot find uuid_generate or uuid_create.  Will use custom uuid generation instead.])
        ;;
    esac
  fi
  BUILD_SCIRUN2="yes"
else
  BUILD_SCIRUN2="no"
fi
AC_SUBST(BUILD_SCIRUN2)


##  ---------------------------------------------------------------------
##  --------------------  Fix Frost config vars  ------------------------
##  ---------------------------------------------------------------------
##  Now that we have run configure with the (xlc,xlC,xlf) compilers, we
##  need to change those to (mpxlc,mpxlC,mpxlf) which automatically use
##  the mpi libraries, so we need to set the MPI variables to be blank.
##  ---------------------------------------------------------------------

  case $host in
    *aix*)
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

AC_CONFIG_FILES(configVars.mk Makefile on-the-fly-libs/Makefile 
                disjointPackageMakefile $sci_defs_files)

##   Only do the following if configuring with Uintah
##   Add a list of Uintah specific output files.
if test "$pkg_Uintah" = "yes" ; then
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


for testfile in $sci_defs_files; do
  file=`echo $testfile | sed 's/test//'`

  if cmp -s $file $testfile 2>/dev/null; then
    AC_MSG_RESULT($file is unchanged.)
    rm -f $testfile
  else
    mv $testfile $file
    AC_MSG_WARN($file has changed.)
  fi
done

#### Clean up: ####
## Possible temporary template directories
rm -rf ii_files
## sedscript
rm -f sedscript
