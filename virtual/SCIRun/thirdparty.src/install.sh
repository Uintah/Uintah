#!/bin/bash

# See README.txt for the arguments to this script.

# 
#  The THIRDPARTY version (used by the tar'ing script).  Please leave in this
#  format:
#
tp_ver='3.0.0'

echo ""
echo "Welcome to the SCIRun Thirdparty Installation Script:"
echo ""

# Note: SHOW_MESSAGE is not a user modifiable variable.
SHOW_MESSAGE=yes

###################
# Removed Packages: dlcompat (no longer needed as it is now built into the Mac)
#                   unp      (never actually integrated into SCIRun)
#                   xerces   (replaced with xml2)
#                   jar      (was used for automatic Insight module wrapping)

### WARNING: This list should reflect the list of PACKAGEs in include_check below.
###    NOTE: There must be an $osname/install-$package.sh file for each of these.
package_list="zlib png freetype Teem mpeg libxml2 tcl tk itcl blt"

# 10 packages must pass before the 3p is considered successfully
# installed.  Do not change this number when testing build of only one
# (or a few) packages.  Otherwise, the script will erroneously report
# a final success if that one (few) package builds.
number_of_packages=10

SH_LIB_EXT="not set, please set me"

INSTALL_DIR="not set, please set me"

output_configure_line()
{
    echo "----------------------------------------------------------"
    echo
    echo "Please use the following to configure SCIRun"
    echo
    echo "cmake ../SCIRun -DSCIRUN_THIRDPARTY_DIR=$full_dir"
    echo
    echo "----------------------------------------------------------"
    echo
}

######################################################
# do_verification is actually used twice (for different stages).
#   The 'Initial verification' stage just checks to see if 
#   (and if so, which parts) of the Thirdparty are already installed.
#   In the 'Final verification' stage, the Thirdparty is checked
#   to see if everything was installed correctly.

do_verification()
{
    if test "$STAGE" = "Initial verification"; then
        echo
        echo "########################################################################"
        echo
        echo "Determining if some thirdparty software has already been installed..."
    else
        echo
        echo "VERIFYING THE INSTALLATION..."
        echo
    fi

    full_dir=$base_dir/$tp_ver/$osname$darwin_proc/$compiler_version-$BITS"bit"

    #
    # Verify installation
    #
    #    The verification is somewhat crude as we are only checking for
    # the existence of certain important files...  In phase one, we check
    # to see what has already been created (and skip it as apparently
    # install is being run again.)

    if test -d "$full_dir/src"; then
        if test "$SHOW_MESSAGE" = "yes"; then
            echo
            echo "It appears that the thirdparty installation was already attempted in this"
            echo "directory.  This script will now attempt to determine which packages were"
            echo "installed correctly.  If a package was already installed, then it will be"
            echo "skipped during this installation.  If you believe that it was NOT"
            echo "installed correctly, you will need to clean things up manually."
            echo "Perhaps the easiest method for the novice is to remove all the contents"
            echo "from the thirdparty dir ($full_dir)"
            echo "and start over."
            echo
        fi

        verify_installation
    else
        if test "$STAGE" = "Initial verification"; then
            echo "  No Thirdparty found. Beginning installation process..."
        fi
    fi
} # end do_verification

check_for_gtar()
{
    tar_cmmd='tar'

    # If the user set the environment variable TAR_CMMD, then use it.
    if test "$TAR_CMMD" != ""; then
        export tar_cmmd=$TAR_CMMD
    fi

    $tar_cmmd --version > /dev/null 2>&1

    if test $? != 0; then
        echo ""
        echo "ERROR: It doesn't appear that 'tar' is a gnu tar."
        echo "Please fix this and then run install.  (You can set"
        echo "the environment variable TAR_CMMD to a gnu tar.)"
        echo ""
        exit
    fi
    if test "$BE_VERBOSE" = "yes"; then
        echo "Tar command is: $tar_cmmd"
    fi
}

setup_num_bits_flags()
{
    ### Calculate the default number of bits on this system...

    BYTES=`echo -e "#include <stdio.h>\n int main(void) { printf(\"%d\",
    sizeof(void *)); return 0; }" | gcc -x c -o bytes - && ./bytes && rm ./bytes`

    if [ $BYTES = "4" ]; then
        BITS="32"
    elif [ $BYTES = "8" ]; then
        BITS="64"
    else
        echo
        echo "ERROR: We were not able to determine the number of bits on your system:"
        echo "       BYTES = $BYTES.  Please contact the scirun-develop@sci.utah.edu mailing list"
        echo "       to resolve this problem."
        echo
        exit
    fi

    echo "Bits: $BITS."
    echo "OS:   $osname."

    if test $BITS = "64" -a "$osname" = "Darwin"; then
        echo
        echo "ERROR: The thirdparty install script does not know how to build a 64 bit OSX build!"
        echo "       Please contact scirun-develop@sci.utah.edu to resolve this issue."
        echo 
        exit
    fi

    ### Setup the flags used to handle various number of bits on various platforms:

    if test "$osname" = "IRIX64"; then
        SH_LIB_EXT="so"
	if test "$BITS" = "64"; then
            BitFlag="-64"
        else 
            BitFlag="-n32"
        fi
    elif test "$osname" = "AIX"; then
        SH_LIB_EXT="a"
	if test "$BITS" = "64"; then
            BitFlag="-q64"
        else
            BitFlag="-q32"
        fi
    elif test "$osname" = "Linux"; then

        SH_LIB_EXT="so"

        ## Check to see if this is a 32 bit box and they want a 64 bit build
        ## or a 64 bit box and they want 32 bits... either way it won't work currently.

        echo ""
        echo "Checking compiler's default word size..."
        cat > /tmp/test_word_size.cc << EOF
        int
        main()
        {
        }
EOF
        $CXX -c /tmp/test_word_size.cc -o /tmp/test_word_size.o
        if test ! -f /tmp/test_word_size.o; then
            echo
            echo "Error in determining compiler's default word size."
            echo "(Please contact dav@sci.utah.edu.)"
            echo "Continuing, but this may cause a problem..."
            sleep 5
            return
        fi
        found_64=`file /tmp/test_word_size.o | grep 64`
        if test -n "$found_64" -a "$BITS" = "32"; then
            echo 
            echo "ERROR:  You have specified a 32 bit build, but the compiler wants to create 64"
            echo "        bit objects by default.  This install script is not smart enough to handle"
            echo "        this.  Please build a 64 bit thirdparty, or contact dav@sci.utah.edu"
            echo "        for more help."
            echo 
            exit
        elif test -z "$found_64" -a "$BITS" = "64"; then
            echo 
            echo "ERROR:  You have specified a 64 bit build, but the compiler wants to create 32"
            echo "        bit objects by default.  This install script is not smart enough to handle"
            echo "        this.  Please build a 32 bit thirdparty, or contact dav@sci.utah.edu"
            echo "        for more help."
            echo 
            exit
        fi
        rm -f /tmp/test_word_size.*

    elif test "$osname" = "Darwin"; then

        SH_LIB_EXT="dylib"

        # Look for the 8 in the 8.0.0 version number to see if we are on 10.4
        os_version=`uname -r | cut -f1 -d"."`
                
	if test "$BITS" = "64"; then
            if test "$os_version" = "7"; then
                echo
                echo "You are not running OSX 10.4... You cannot use 64 bit mode.  Exiting..."
                echo
                exit
            fi
        fi
    fi
}

check_for_gmake()
{
    gmake --version > /dev/null 2>&1
    if test $? = 0; then
        MAKE='gmake '
    else
        make --version > /dev/null 2>&1

        if test $? = 0; then
            MAKE='make '
        else
            echo "Couldn't find a suitable 'make'.  Exiting."
            exit
        fi
    fi
    if test "$BE_VERBOSE" = "yes"; then
        echo "Make program is: $MAKE"
    fi
    export MAKE
}

determine_compiler_version()
{
    if test "$osname" = "Linux" || test "$osname" = "Darwin"; then	
    	# In case $CC is a full path, cut off the path so only the name
    	# of the compiler executable remains
    	if test "${CC:0:1}" = "/"; then
    		# $CC is a full path
    	    pathlength=`echo $CC | grep -o "/" | wc -l | sed s/\ //g`
    	   	pathlength=`expr $pathlength + 1` # Since cut will create an empty 
    			                          # field before the initial '/'
    		cc_exec=`echo $CC | cut -f $pathlength -d '/'`	
    	else
    		cc_exec=$CC
    	fi
        number=`$CC --version | head -1 | cut -f3 -d" "`
    	compiler_version="$cc_exec-$number"
    elif test "$osname" = "IRIX64"; then
        # 2>&1 redirects stderr to stdout
        number=`$CC -version 2>&1 | head -1 | cut -f4 -d" "`
        compiler_version="MIPSpro-$number"
    elif test "$osname" = "AIX"; then
        number=`what /usr/bin/xlc | grep Version | head -1 | awk '{print $2}'  | awk -F. '{ print $1 "." $2 }'`
        echo "number is '$number'"
        compiler_version="xlc-$number"
    else
        compiler_version="Unknown"
    fi
    if test "$BE_VERBOSE" = "yes"; then
        echo "Compiler ($CC) version: $compiler_version"
    fi
}

determine_machine_info()
{
    processor=`uname -m`
    osname=`uname`

    if test "$BE_VERBOSE" = "yes"; then
        echo "Processor: $processor."
        echo "OS: $osname"
    fi

    if test "$BE_VERBOSE" = "yes"; then
        if test -n "$CC"; then
            echo "Using C compiler $CC from environment variable CC"
        fi
        if test -n "$CXX"; then
            echo "Using CXX compiler $CXX from environment variable CXX"
        fi
    fi

    shared_libs="DEFAULT"

    if test "$osname" = "Darwin"; then
       # Figure out if it is a ppc or an intel processor
       darwin_proc="-`uname -p`"
    fi

    # AIX
    if test "$osname" = "AIX"; then
        shared_libs="DISABLE"
        export LDR_CNTRL="LARGE_PAGE_DATA=Y"
        if test -z "$CC" || test -z "$CXX"; then
            echo
            echo "Please set the CC and CXX environment variable!"
            echo
        fi

    # LINUX
    elif test "$osname" = "Linux" || test "$osname" = "Darwin"; then
        if test -z "$CC"; then
            give_cc_msg=1
            export CC=gcc
        fi
        if test -z "$CXX"; then
            give_cxx_msg=1
            export CXX=g++
        fi

    # IRIX
    else
        if test -z "$CC"; then
            give_cc_msg=1
            export CC=cc
        fi
        if test -z "$CXX"; then
            give_cxx_msg=1
            export CXX=CC
        fi
    fi
    if test "$BE_VERBOSE" = "yes"; then
        if test -n "$give_cc_msg"; then
            echo "Using C compiler: $CC"
        fi
        if test -n "$give_cxx_msg"; then
            echo "Using C++ compiler: $CXX"
        fi
    fi
}


show_usage () 
{
    if test -f "README.txt"; then
        cat README.txt
    else
        echo ""
        echo "WARNING: README.txt is missing!"
        echo ""
        echo "The usage information is contained in the README.txt file."
        echo "Unfortunately, this file does not seem to exist.  Please"
        echo "check that your installation directory is instact."
        echo
    fi
    exit
}

parse_args()
{
    ### Optional Arg 1: --uintah-only
    ### This argument will cause only the third party packages that Uintah depends on
    ### to be built.  Currently, that means Teem and libxml2.
    if test "$1" = "--uintah-only"; then
        echo "Building only the packages necessary for Uintah."
        package_list="libxml2"
        shift 1
    fi

    if test "$1" = "--seg3d-only"; then
        echo "Building only the packages necessary for Seg3D."
        package_list="zlib png freetype Teem mpeg libxml2"
        number_of_packages=6
        seg3d=1
        shift 1
    fi

    ### Arg 1: base directory
    INSTALL_DIR=$1
    if [ -z "$INSTALL_DIR" ]; then
        echo ""
        echo "ERROR: You must specify an installation directory."
        show_usage
    else
        # Check to see if the dir name ends with a '/'.  If so, remove it.
        good_dir=`echo $INSTALL_DIR | sed "s%.*/%%g"`
        if test -z $good_dir; then
            # Remove the / from the end of the line.
            INSTALL_DIR=`echo $INSTALL_DIR | sed "s%/$%%"`
        fi
    fi

    if [ ! -d "$INSTALL_DIR" ]; then
        echo
        echo "ERROR: The directory ($INSTALL_DIR) you specified does not exist."
        echo "       Please create the directory first and then run the install again."
        show_usage
    fi

    base_dir=$INSTALL_DIR

    ### Arg 2: make flags
    if [ ! -z $2 ]; then
        make_flags="$2"
        if test `echo $make_flags | cut -c1` != "-"; then
            if test "$make_flags" = "32" -o "$make_flags" = "64"; then
                echo
                echo "The number of bits (32|64) is no longer required as an argument."
                echo "Please run the install command again without it."
                echo
            else
                echo
                echo "The argument ($2) is invalid.  It should be something like '-j#' where # is"
                echo "the number of processors on your system.  Please fix this and run the install"
                echo "command again."
                echo
            fi
            exit
        fi
    fi

}

pre_installation_checks() {
    #
    # PRE-INSTALLATION CHECKS:
    #
    # Check for things that may be needed by various installs so that
    # a heads up can be given to the user before the installation
    # begins such that the user can interrupt the installation at this
    # point to take care of things instead of having to interrupt it
    # during the middle and waste a lot of time.
    #

    echo
    echo "Pre-install checks...  NONE CURRENTLY NEEDED"
    echo
}

check_for_libxml2()
{
    ## The standard distributed version of libxml2 is still broken (at
    ## least on the Mac, SGI, Windows), so don't give the user the
    ## option of using it.  Once the libxml2 people update their code
    ## and we think it is in use by most people, then remove this
    ## comment and the following return statement.
    return

    ###############
    # Check to see if libxml2 already exists on the system.

    if test "$BE_VERBOSE" = "yes"; then
        echo "########################################################################"
        echo "Testing for system libxml2."
    fi
    
    xml2config=`which xml2-config`

    if test -z "$xml2config"; then
        echo
        echo "Failed to find a system libxml2 of at least version 2.6. (No 'xml2-config'"
        echo "program found.)  Will therefore compile libxml2 into the Thirdparty."
        echo
        return
    else
        # figure out if libxml2 is new enough
        major=`xml2-config --version | awk -F. '{ print $1 }'`
        minor=`xml2-config --version | awk -F. '{ print $2 }'`
        if test "$major" -eq 2 -a "$minor" -lt 6  || test "$major" -lt 2 ; then
            echo
            echo "System libxml (version $major.$minor) is not new enough (need at least 2.6)."
            echo "Will therefore compile libxml2 into the Thirdparty."
            echo
            return
        fi
    fi

    echo
    echo "A sufficiently new libxml2 was found (in `xml2-config --prefix`/lib).  You may choose to use"
    echo "this libxml2 instead of having the SCIRun Thirdparty build one for you."
    echo
    answer=""
    while [[ $answer != "y" && $answer != "n" ]]; do
        echo "Do you wish to use the system libxml2 instead of building one in the Thirdparty? (y/n)"
        echo
        read answer
    done
    if test "$answer" = "y"; then
        DO_NOT_BUILD_LIBXML2=true
        libxml2="Using Built-In System Libraries"
        number_of_packages=`echo "$number_of_packages - 1" | bc`
    else
	echo 
	echo "Ok, I'll build it for you."
    fi
    
} # end check_for_libxml2

check_for_utility_progs()
{
    DIRNAME=`which dirname`
    if test -z "$DIRNAME"; then
        echo
        echo "'dirname' command not found.  This command is necessary for this"
        echo "install script.  Please contact your system administrator to have"
        echo "this fixed.  Bye."
        echo
        exit
    fi
    BASENAME=`which basename`
    if test -z "$BASENAME"; then
        echo
        echo "'basename' command not found.  This command is necessary for this"
        echo "install script.  Please contact your system administrator to have"
        echo "this fixed.  Bye."
        echo
        exit
    fi
}

find_system_libs()
{
    # Determine if certain 3rd Party libs already exist on the system.
    # If so, ask the user if they wish to use the already installed
    # version, or to instlal the SCIRun version.

    check_for_utility_progs

    # Only check for system 
    if test "$libxml2" = "not_installed"; then
        check_for_libxml2
    fi
}

##########################################################################################
#
# The following list contains all the packages that are to be installed
# and the files that should have been installed with that package.
#
# To add more check information to 'include_check' use this format:
# PACKAGE <name> INCLUDES <list of files that must be found in include directory>
# LIBS <list of files that must be found in lib directory> BIN <list of files
# that must be found in the bin directory> DONE
#
# INCLUDES, LIBS, and BIN are all optional.  DONE must be the last
# word in the list.  This is all one big string, so make sure to use the "\"
# to continue each line.

### !!!WARNING!!!: PACKAGE name (eg: tcl) cannot have a '.' in it... (best if it is all letters).

add_freetype_packages()
{
    FREETYPE_LIST="PACKAGE freetype                                \
                           BIN          freetype-config            \
                           INCLUDES     ft2build.h freetype2/freetype/freetype.h freetype2/freetype/ftbbox.h \
                                        freetype2/freetype/ftbdf.h freetype2/freetype/ftbitmap.h \
                                        freetype2/freetype/ftcache.h freetype2/freetype/ftchapters.h \
                                        freetype2/freetype/fterrdef.h freetype2/freetype/fterrors.h \
                                        freetype2/freetype/ftglyph.h freetype2/freetype/ftgzip.h \
                                        freetype2/freetype/ftimage.h freetype2/freetype/ftincrem.h \
                                        freetype2/freetype/ftlist.h freetype2/freetype/ftlzw.h \
                                        freetype2/freetype/ftmac.h freetype2/freetype/ftmm.h \
                                        freetype2/freetype/ftmodapi.h freetype2/freetype/ftmoderr.h \
                                        freetype2/freetype/ftotval.h freetype2/freetype/ftoutln.h \
                                        freetype2/freetype/ftpfr.h freetype2/freetype/ftrender.h \
                                        freetype2/freetype/ftsizes.h freetype2/freetype/ftsnames.h \
                                        freetype2/freetype/ftstroke.h freetype2/freetype/ftsynth.h \
                                        freetype2/freetype/ftsysio.h freetype2/freetype/ftsysmem.h \
                                        freetype2/freetype/ftsystem.h freetype2/freetype/fttrigon.h \
                                        freetype2/freetype/fttypes.h freetype2/freetype/ftwinfnt.h \
                                        freetype2/freetype/ftxf86.h freetype2/freetype/internal/autohint.h \
                                        freetype2/freetype/internal/tttypes.h \
                                        freetype2/freetype/t1tables.h freetype2/freetype/ttnameid.h \
                                        freetype2/freetype/tttables.h freetype2/freetype/tttags.h \
                                        freetype2/freetype/ttunpat.h \
                           LIBS         libfreetype.$SH_LIB_EXT libfreetype.a"
}

add_tcl_packages()
{
    TCL_LIST="PACKAGE tcl                                \
                           INCLUDES     tcl.h tclDecls.h tclInt.h tclIntDecls.h tclPlatDecls.h               \
                           LIBS         libtcl.$SH_LIB_EXT libtcl8.3.$SH_LIB_EXT libtclstub8.3.a tcl tcl8.3 tclConfig.sh"
}

add_tk_packages()
{
    TK_LIST="PACKAGE tk                                 \
                          INCLUDES     tk.h tkDecls.h tkIntXlibDecls.h tkPlatDecls.h                        \
                          LIBS         libtk.$SH_LIB_EXT libtk8.3.$SH_LIB_EXT libtkstub8.3.a tk tk8.3 tkConfig.sh"
}

add_itcl_packages()
{
    ITCL_LIST="PACKAGE itcl                               \
                            INCLUDES     itcl.h itclDecls.h itclInt.h itclIntDecls.h itkDecls.h               \
                            LIBS         itcl itcl3.1 itclConfig.sh itk iwidgets iwidgets2.2.0 iwidgets3.1.0  \
                                         libitcl.$SH_LIB_EXT libitcl3.1.$SH_LIB_EXT libitclstub3.1.a libitk.$SH_LIB_EXT"
}

add_blt_packages()
{
    BLT_LIST="PACKAGE blt                                \
                            INCLUDES     blt.h bltBind.h bltChain.h bltHash.h bltList.h bltPool.h bltTree.h bltVector.h   \
                            LIBS         blt2.4 libBLT.a libBLT.$SH_LIB_EXT libBLT24.a libBLT24.$SH_LIB_EXT libBLTlite.a  \
                                         libBLTlite24.a libBLTlite24.$SH_LIB_EXT"
}

add_mpeg_packages()
{
    MPEG_LIST="PACKAGE mpeg                               \
                            INCLUDES     mpege.h mpege_im.h    \
                            LIBS         libmpege.a"
}

add_teem_packages()
{
    TEEM_LIST="PACKAGE Teem                               \
                            BIN          affine airSanity cubic emap gkms idx2pos ilk miter mrender           \
                                         nrrdSanity overrgb pos2idx pprobe qbert spots tend undos ungantry    \
                                         unu vprobe            \
                            INCLUDES                           \
                                         teem/air.h teem/biff.h teem/mite.h teem/nrrd.h teem/ten.h teem/unrrdu.h  \
                            LIBS                               \
                                         libair.a libalan.a libbane.a libbiff.a libdye.a libecho.a libell.a   \
                                         libgage.a libhest.a libhoover.a liblimn.a libmite.a libmoss.a        \
                                         libnrrd.a libteem.a libten.a libunrrdu.a"
}


add_libxml2_packages()
{
     LIBXML2_LIST="PACKAGE libxml2                        \
                       BIN          xml2-config xmlcatalog xmllint \
                       INCLUDES     libxml2/libxml/c14n.h libxml2/libxml/catalog.h libxml2/libxml/chvalid.h \
                                    libxml2/libxml/debugXML.h libxml2/libxml/dict.h libxml2/libxml/DOCBparser.h \
                                    libxml2/libxml/encoding.h libxml2/libxml/entities.h libxml2/libxml/globals.h \
                                    libxml2/libxml/hash.h libxml2/libxml/HTMLparser.h libxml2/libxml/HTMLtree.h \
                                    libxml2/libxml/list.h libxml2/libxml/nanoftp.h \
                                    libxml2/libxml/nanohttp.h libxml2/libxml/parser.h libxml2/libxml/parserInternals.h \
                                    libxml2/libxml/pattern.h libxml2/libxml/relaxng.h libxml2/libxml/SAX2.h libxml2/libxml/SAX.h \
                                    libxml2/libxml/schemasInternals.h libxml2/libxml/schematron.h libxml2/libxml/threads.h \
                                    libxml2/libxml/tree.h libxml2/libxml/uri.h libxml2/libxml/valid.h \
                                    libxml2/libxml/xinclude.h libxml2/libxml/xlink.h libxml2/libxml/xmlautomata.h \
                                    libxml2/libxml/xmlerror.h libxml2/libxml/xmlexports.h libxml2/libxml/xmlIO.h \
                                    libxml2/libxml/xmlmemory.h libxml2/libxml/xmlmodule.h libxml2/libxml/xmlreader.h \
                                    libxml2/libxml/xmlregexp.h libxml2/libxml/xmlsave.h libxml2/libxml/xmlschemas.h \
                                    libxml2/libxml/xmlschemastypes.h libxml2/libxml/xmlstring.h libxml2/libxml/xmlunicode.h \
                                    libxml2/libxml/xmlversion.h libxml2/libxml/xmlwriter.h \
                                    libxml2/libxml/xpath.h libxml2/libxml/xpathInternals.h libxml2/libxml/xpointer.h \
                       LIBS         libxml2.a"
}


add_png_packages()
{
    PNG_LIST="PACKAGE png                               \
                            INCLUDES     png.h pngconf.h libpng12/png.h libpng12/png.h \
                            LIBS         libpng.$SH_LIB_EXT libpng.a"
}

add_zlib_packages()
{
    ZLIB_LIST="PACKAGE zlib                               \
                            INCLUDES     zlib.h zconf.h   \
                            LIBS         libz.a"
    if test "$shared_libs" = "DEFAULT"; then
         if test "$osname" = "Darwin"; then
            ZLIB_LIST="$ZLIB_LIST libz.1.2.3.$SH_LIB_EXT"
         else
            ZLIB_LIST="$ZLIB_LIST libz.$SH_LIB_EXT.1.2.3"
         fi
    fi
}

verify_installation()
{
    all_packages=

    for package in $package_list; do

        if test "$package" = "tcl"; then
            add_tcl_packages
        elif test "$package" = "tk"; then
            add_tk_packages
        elif test "$package" = "itcl"; then
            add_itcl_packages
        elif test "$package" = "blt"; then
            add_blt_packages
        elif test "$package" = "freetype"; then
            add_freetype_packages
        elif test "$package" = "mpeg"; then
            add_mpeg_packages
        elif test "$package" = "Teem"; then
            add_teem_packages
        elif test "$package" = "png"; then
            add_png_packages
        elif test "$package" = "zlib"; then
            add_zlib_packages
        elif test "$package" = "libxml2"; then
            if test "$DO_NOT_BUILD_LIBXML2" = "true"; then
                LIBXML2_LIST=
                all_packages=libxml2
            else
                add_libxml2_packages
            fi
        fi
    done

    include_check="$FREETYPE_LIST $TCL_LIST $TK_LIST $ITCL_LIST $BLT_LIST $LIBXML2_LIST $MPEG_LIST $TEEM_LIST $PNG_LIST $ZLIB_LIST DONE"

    # Tracks whether anything fails.
    all_passed=yes

    # Used to specify printing out the name of the Package
    show_name=no

    # Tracks whether any file in the package failed.
    package_good=yes

    # First time through entire loop.
    first_time=yes
    num_passed=0

    for file in $include_check; do

        if test "$show_name" = "yes"; then
            all_packages="$all_packages $file"
            package_name=$file
            first_time=no
            show_name=no
            continue
        fi
        if test "$file" = "PACKAGE"; then
            if test "$first_time" = "no"; then
                if test "$package_good" = "yes"; then
                    num_passed=`echo "$num_passed + 1" | bc`
                    eval $package_name=installed
                fi
            fi
            package_good=yes
            show_name=yes
            continue
        fi

        if test "$file" = "DONE"; then
            # Last time through, print out status of the last package.
            if test "$package_good" = "yes"; then
                num_passed=`echo "$num_passed + 1" | bc`
                eval $package_name=installed
            fi
            # then drop out of loop.
            continue
        fi
        if test "$file" = "INCLUDES"; then
            dir="include"
            continue
        fi

        if test "$file" = "LIBS"; then
            dir="lib"
            continue
        fi

        if test "$file" = "BIN"; then
            dir="bin"
            continue
        fi

        if test ! -e $full_dir/$dir/$file; then
            all_passed=no
            package_good=no
        fi
 
    done

    # Used to specify printing out the name of the Package
    show_name=no

    # Tracks whether any file in the package failed.
    package_good=yes

    # First time through entire loop.
    first_time=yes

    if test "$BE_VERBOSE" = "yes" && test $all_passed = "no"; then
        echo "Specific (file by file) results:"
        echo "--------------------------------"

        for file in $include_check; do

            if test "$show_name" = "yes"; then
                echo "   PACKAGE: $file"
                first_time=no
                show_name=no
                continue
            fi

            if test "$file" = "PACKAGE"; then

                if test "$first_time" = "no"; then
                    if test "$package_good" = "no"; then
                        echo "   FAILED"
                    else
                        echo "   SUCCEEDED"
                    fi
                    echo ""
                fi
                package_good=yes
                show_name=yes
                continue
            fi

            if test "$file" = "DONE"; then
            # Last time through, print out status of the last package.
                if test "$package_good" = "no"; then
                    echo "   FAILED"
                else
                    echo "   SUCCEEDED"
                fi
                echo ""
                # then drop out of loop.
                continue
            fi

            if test "$file" = "INCLUDES"; then
                dir="include"
                continue
            fi

            if test "$file" = "LIBS"; then
                dir="lib"
                continue
            fi

            if test "$file" = "BIN"; then
                dir="bin"
                continue
            fi

            if test -e $full_dir/$dir/$file; then
                echo "      $file: yes"
            else
                all_passed=no
                package_good=no
                echo "      $file: no"
            fi
        done
    fi

    echo "$STAGE results:"
    echo "------------------------------"
    for package in $package_list; do
        echo "$package:		${!package}"
    done
    echo

    if test "$all_passed" = "no"; then
        echo "WARNING: One or more thirdparty packages did not install correctly!"
        echo ""
        # If this is the first time through, check to see if one of the tarballs
        # has already been un-tar'ed.  If so, ask the user if they would like to
        # NOT un-tar the files again.  
        if test "$STAGE" = "Initial verification" && test -d $full_dir/src/teem-1.9.0-src; then
            echo "It appears that some source tarballs have already been un-tar'ed."
            echo "Do you wish to skip un-tar'ing (again) all sources? (y/n)"
            echo "  If you choose 'y', then no new un-tar'ing will occur."
            echo "  If you choose 'n', then you will be asked for each library."
            read answer
            echo
            if test "$answer" = "y"; then
                export DO_NOT_UNTAR=yes
            fi
        fi

    elif test "$STAGE" = "Initial verification" -a "$num_passed" -eq "$number_of_packages" ; then
        # We are done...
        echo "It appears that the Thirdparty is already installed."
        echo
        output_configure_line
        exit
    elif test "$num_passed" -eq "$number_of_packages"; then
        # Make sure permissions are set such that we can fix things
        # where necessary.
        echo "Updating permissions on thirdparty files (go+rX)."
        echo "   bin..."
        chmod -R u+w,go+rX $full_dir/bin > /dev/null 2>&1
        echo "   include..."
        chmod -R u+w,go+rX $full_dir/include > /dev/null 2>&1
        echo "   lib..."
        chmod -R u+w,go+rX $full_dir/lib > /dev/null 2>&1
        echo "   man..."
        chmod -R u+w,go+rX $full_dir/man > /dev/null 2>&1
        echo "   share..."
        chmod -R u+w,go+rX $full_dir/share > /dev/null 2>&1
        echo "   src..."
        chmod -R u+w,go+rX $full_dir/src > /dev/null 2>&1
        output_configure_line
    else
        echo "WARNING:"
        echo "--------"
        echo "   All specified packages passed, but not all of the thirdparty was specified..."
        echo "   Thus the Thirdparty is not yet ready for use with SCIRun."
        echo
        exit
    fi

    echo ""

} # END verify_installation()


#####################################################################
# Create the src directory (where the tarballs will be un-tar'ed into).

build_src_dir()
{
    if test ! -d $full_dir/src; then
        if test "$BE_VERBOSE"; then
            echo "mkdir -p $full_dir/src"
            echo
        fi
        mkdir -p $full_dir/src

        # Make the version file here.
        if test ! -f "$full_dir/SCIRUN_THIRDPARTY_VERSION"; then 
           echo "SCIRUN_THIRDPARTY_VERSION=$tp_ver" > $full_dir/SCIRUN_THIRDPARTY_VERSION
           chmod go+r $full_dir/SCIRUN_THIRDPARTY_VERSION
        fi

        # Also make the lib and include dir as some packages (eg:
        # mpeg) don't seem to be able to do this for themselves.
        mkdir $full_dir/lib
        mkdir $full_dir/include

        echo ""
        echo "Updating permissions on $base_dir/*/*/* (go+rX)"
        echo ""
        chmod u+w,go+rX $base_dir/* > /dev/null 2>&1
        chmod u+w,go+rX $base_dir/*/* > /dev/null 2>&1
        chmod u+w,go+rX $base_dir/*/*/* > /dev/null 2>&1
    fi
}

#####################################################################
# Save the install line to a install_command.txt file for future
# reference.

save_install_line()
{
    echo ""
    echo "########################################################################"
    echo ""
    echo "Saving install line to install_command.txt for future reference."
    echo "$0 $*" > install_command.txt
    echo "SCIRUN_THIRDPARTY_DIR=${full_dir}/" >> install_command.txt
    echo ""
}

#####################################################################
# Install required packages:

install()
{
    # Get the system path of bash for use in running each install script
    bash_path=`which bash`

    echo "########################################################################"
    echo
    echo "Installing in: $full_dir"
    
    for package in $package_list; do

        echo    ""
        echo    "########################################################################"
        echo    ""
        echo -n "Installing '$package'..."
        echo    ""

        package_value=${!package}

        if test "$package_value" = "not_installed"; then
            install_file=Scripts/install-$package.sh
            if test ! -f "$install_file"; then
                echo
                echo "ERROR in install.sh script. $install_file does not exist."
                echo "Report this to scirun-develop@sci.utah.edu."
                echo
                exit
            fi
            $bash_path $install_file $full_dir $osname $SH_LIB_EXT $BITS $tar_cmmd $shared_libs $make_flags
        else
            echo "	already installed... skipping."
        fi
    done

} # end install

##############################
############ MAIN ############

BE_VERBOSE=yes

##### CHECK FOR BASICS #######

determine_machine_info  # Must be first call (to get 'osname' set).
parse_args $*
check_for_gtar
check_for_gmake
determine_compiler_version
setup_num_bits_flags

# Initialize packages to not_installed
for package in $package_list; do
    eval $package=not_installed
done

########  DETERMINE WHAT NEEDS TO BE INSTALLED... ###

STAGE="Initial verification"
do_verification

######## Determine if the system already has libxml2, etc.

find_system_libs

######## Create the directory to put the thirdpary in.

build_src_dir

######## SAVE INSTALL LINE ##############

save_install_line $*

######## INSTALL ##############

install

######## VERIFY INSTALL #######

STAGE="Final verification"
SHOW_MESSAGE=no
do_verification

######## Remove Shared Libs #######
#
# This is done so that SCIRun applications (such as Seg3D) can be
# linked staticly.  However, it probably would be better to also
# update the verification step to only look for the .a's (instead of
# doing the removal after the verification).  Also, we probably should
# update each of the build scripts to only build staticly if we only
# want static libs.  Finally, in theory we should be able to use a 
# -static flag when linking the SCIRun Apps and thus the linker will
# only use the .a's even when .so's exist.
#
if [ ! -z "$seg3d" ]; then
    cd ${full_dir}/lib/
    rm -rf *.${SH_LIB_EXT}*
fi

