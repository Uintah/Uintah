# ExprLibConfig.cmake - defines variables to use the expression
#  library in another cmake project.  The following variables are
#  defined:
#
# ExprLib_FOUND - true (1)
# 
# ExprLib_INCLUDE_DIR - the include directory for the installed header files
#
# ExprLib_ENABLE_THREADS - boolean set to true if multicore support is enabled
# ExprLib_NTHREADS - number of threads that the expression library supports.
#
# typical usage:
#   find_package( ExprLib )
#   include_directories_( ${ExprLib_INCLUDE_DIR} )
#   target_link_libraries( ... exprlib )
#

set( ExprLib_INCLUDE_DIR /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/ExprLib/include /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/SpatialOps/include;/usr/local/boost/include;/usr/local/boost/include;/home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/SpatialOps/include;/usr/local/boost/include )
set( ExprLib_TPL_LIBS spatialops;/usr/local/boost/lib/libboost_program_options.so;/usr/local/boost/lib/libboost_system.so )

find_package( SpatialOps PATHS /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/SpatialOps/share )
include( /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/ExprLib/lib/expression/ExprLib.cmake )

set( ExprLib_ENABLE_OUTPUT OFF )
set( ExprLib_ENABLE_THREADS OFF )
set( ExprLib_NTHREADS 1 )
set( ExprLib_ENABLE_LBMS OFF )
set( ExprLib_ENABLE_UINTAH ON )

set( ExprLib_COMPILER_DEFS -DENABLE_UINTAH )

foreach( prop ${ExprLib_COMPILER_DEFS} )
  add_definitions( ${prop} )
endforeach()

if( "${ExprLib_ENABLE_OUTPUT}" )
  set( ExprLib_compare_database /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/ExprLib/bin/compare_database )
endif()

set( ExprLib_create_expr /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/ExprLib/bin/create_expr )
