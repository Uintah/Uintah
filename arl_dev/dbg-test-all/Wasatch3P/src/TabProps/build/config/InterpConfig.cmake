# InterpConfig.cmake - defines variables to use the linear interpolants
#  library in another cmake project.  The following variables are
#  defined:
#
# Interp_FOUND - true (1)
# 
# Interp_INCLUDE_DIR - the include directory for the installed header files
#
# typical usage:
#   find_package( Interp )
#   include_directories_( ${Interp_INCLUDE_DIR} )
#   target_link_libraries( ... linear_interp )
#

include( /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/TabProps/lib/tabprops/Interpolant.cmake )
set( Interp_INCLUDE_DIR /home/alan/uintah-md/md-dev/dbg-test-all/Wasatch3P/install/TabProps/include /usr/local/boost/include )
set( Interp_FOUND 1 )
