cmake_minimum_required(VERSION 2.8.4)
project(TabProps)

find_package( Git )
if( GIT_FOUND )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 "--pretty=format:\"%H\""
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE TABPROPS_REPO_HASH 
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 "--pretty=format:\"%cd\""
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE TABPROPS_REPO_DATE 
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
  set( TABPROPS_REPO_DATE "\"No information available\"" )
  set( TABPROPS_REPO_HASH "\"No information available\"" )
endif()

set( MAJOR_VERSION "1" )
set( MINOR_VERSION "1" )
set( PATCH_VERSION "0" )

#-------------------------------------
# set some configuration options
option( Cantera_CMAKE "Cantera was built with CMake" ON )
option( TabProps_PREPROCESSOR "enable building of preprocessor tools for TabProps (requires Cantera)" ON )
option( TabProps_UTILS "enable building tabprops utilities" ON )
option( TabProps_ENABLE_TESTING "enable regression tests)" ON )

# default to release builds
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE )
endif( NOT CMAKE_BUILD_TYPE )

# set the default install dir to the local build dir (instead of the typical /usr/bin)
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX
    ${CMAKE_CURRENT_BINARY_DIR}
    CACHE PATH "Installation directory" FORCE
    )
endif()

# this is where the configuration files for downstream cmake-enabled usage of TabProps
# this is typically in some "share" directory like /usr/local/share
set( TabProps_CONFIG_INSTALL
  ${CMAKE_INSTALL_PREFIX}/share
  CACHE PATH
  "Location to install configuration files"
  )

if( APPLE )
  message( STATUS "C++ version: " ${CMAKE_CXX_COMPILER_VERSION} )
  if( NOT ${CMAKE_CXX_COMPILER_VERSION} LESS "5.0" )
    # on mac with boost 1.52 and xcode 5, the default template
    # expansion depth was exceeded.  This fixed the problem:
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=256" )
    message( STATUS "Increasing template recursion depth" )
  endif()
endif( APPLE )

#-------------------------------------
configure_file(
  TabPropsConfig.h.in
  ${PROJECT_BINARY_DIR}/config/tabprops/TabPropsConfig.h
  @ONLY
  )
install( FILES
  ${PROJECT_BINARY_DIR}/config/tabprops/TabPropsConfig.h
  DESTINATION include/tabprops
  )
include_directories( ${PROJECT_BINARY_DIR}/config )  



#--------------------------------------
# resolve required libraries

if( TabProps_PREPROCESSOR )
  # cantera
  if( Cantera_CMAKE )
    find_package( Cantera PATHS ${Cantera_DIR} ${Cantera_DIR}/share )
  else()
    set( Cantera_DIR "" CACHE PATH "Location where cantera was installed" )
    set( Cantera_BIN_DIR ${Cantera_DIR}/bin )
    set( Cantera_INCLUDE_DIR ${Cantera_DIR}/include )
    set( Cantera_LIBRARY ${Cantera_DIR}/lib )
    set( Cantera_LIBRARIES
      ${Cantera_LIBRARY}/libctcxx.a
      ${Cantera_LIBRARY}/libequil.a
      ${Cantera_LIBRARY}/libtransport.a
      ${Cantera_LIBRARY}/libtpx.a
      ${Cantera_LIBRARY}/libconverters.a
      ${Cantera_LIBRARY}/libuser.a
      ${Cantera_LIBRARY}/libctnumerics.a
      ${Cantera_LIBRARY}/libkinetics.a
      ${Cantera_LIBRARY}/libctf2c.a
      ${Cantera_LIBRARY}/libctbase.a
      ${Cantera_LIBRARY}/libctmath.a
      ${Cantera_LIBRARY}/libthermo.a
      ${Cantera_LIBRARY}/libtpx.a
      # this may cause trouble:
      blas
      lapack
      )
  endif( Cantera_CMAKE )
  set( TPL_LIBRARIES ${TPL_LIBRARIES} ${Cantera_LIBRARIES} )
  set( TPL_INCLUDE ${TPL_INCLUDE} ${Cantera_INCLUDE_DIR} )
endif( TabProps_PREPROCESSOR )

#-- Boost
# SpatialOps will set information about where it found the Boost libraries so that
# we can pick that up here and look for additional boost libraries to build against.
set(Boost_ADDITIONAL_VERSIONS ${Boost_ADDITIONAL_VERSIONS}
    "1.49" "1.49.0" "1.50" "1.50.0" "1.51" "1.51.0" "1.52" "1.52.0"
    "1.53" "1.53.0" "1.54" "1.54.0" "1.55" "1.55.0" )
set(Boost_USE_MULTITHREAD ON)
set( boost_comps ${boost_comps} "serialization" )
if( TabProps_ENABLE_TESTING )
  set( boost_comps ${boost_comps} "program_options;chrono;system" )
endif( TabProps_ENABLE_TESTING )
if( TabProps_PREPROCESSOR OR TabProps_UTILS )
  set( boost_comps ${boost_comps} "regex;filesystem;system;program_options" )
endif( TabProps_PREPROCESSOR OR TabProps_UTILS )

if( DEFINED BOOST_ROOT )
  message( STATUS "Setting boost installation location as: " ${BOOST_ROOT} )
  set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
endif( DEFINED BOOST_ROOT )
if( DEFINED BOOST_INCLUDEDIR AND DEFINED BOOST_LIBRARYDIR )
  message( STATUS "Setting boost include location as: " ${BOOST_INCLUDEDIR} )
  message( STATUS "Setting boost library location as: " ${BOOST_LIBRARYDIR} )
  set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
endif()
message( STATUS "Looking for boost libs: '${boost_comps}'" )
set(BOOST_MIN_VERSION "1.49.0" )
find_package( Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS ${boost_comps} )
message( STATUS "Boost header installation found at: " ${Boost_INCLUDE_DIR} )
message( STATUS "      library installation found at: " ${Boost_LIBRARY_DIRS} )
message( STATUS "      version: " ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION} )
message( STATUS " found boost libs: '${Boost_LIBRARIES}'" )

set( TPL_INCLUDE ${TPL_INCLUDE} ${Boost_INCLUDE_DIR} )
set( TPL_LIBRARIES ${TPL_LIBRARIES} ${Boost_LIBRARIES} )

#--------------------------------------

include_directories( . ${TPL_INCLUDE} )

if( TabProps_ENABLE_TESTING )
  enable_testing()
endif( TabProps_ENABLE_TESTING )
add_subdirectory( tabprops )
if( TabProps_ENABLE_TESTING )
  add_subdirectory( test )
endif()

#-----------------------------------------------------------
#----------------------- INSTALLATION ----------------------
message( STATUS "Files will be installed to: " ${CMAKE_INSTALL_PREFIX}/tabprops )


#if( TabProps_PREPROCESSOR )
#
#   install( FILES ${Cantera_BIN_DIR}/ck2cti
#     ${Cantera}/bin/cti2ctml
#     ${Cantera}/data/elements.xml
#     DESTINATION bin
#     )
#
#endif( TabProps_PREPROCESSOR )

configure_file( ${PROJECT_SOURCE_DIR}/TabPropsConfig.cmake.in
  ${PROJECT_BINARY_DIR}/config/TabPropsConfig.cmake
  @ONLY
  )
install( FILES
  ${PROJECT_BINARY_DIR}/config/TabPropsConfig.cmake
  DESTINATION ${TabProps_CONFIG_INSTALL}
  )
#------------------------------------------------------------


# Information for creating packages
set( CPACK_PACKAGE_VENDOR "James C. Sutherland (james.sutherland@utah.edu" )
set( CPACK_PACKAGE_VERSION_MAJOR ${MAJOR_VERSION} )
set( CPACK_PACKAGE_VERSION_MINOR ${MINOR_VERSION} )
set( CPACK_PACKAGE_VERSION_PATCH ${PATCH_VERSION} )
include( CPack )
