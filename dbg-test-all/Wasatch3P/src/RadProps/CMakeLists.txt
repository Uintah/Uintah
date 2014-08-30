cmake_minimum_required(VERSION 2.8.4)

project(RadProps)

option( RadProps_ENABLE_TESTING "enable regression tests" ON )
option( RadProps_ENABLE_PREPROCESSOR "enable preprocessing and table querying utilities" ON )
set( TabProps_DIR "" CACHE PATH "Path to installation of SpatialOps" )

# Set the location for the installation.  Comment this out to install
# to /usr/local.  Or modify it to install to a different location.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set( CMAKE_INSTALL_PREFIX
    ${CMAKE_CURRENT_BINARY_DIR}
    CACHE PATH "Installation directory" FORCE
    )
endif()

# default to release builds
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE )
endif( NOT CMAKE_BUILD_TYPE )

find_package( Git )
if( GIT_FOUND )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 "--pretty=format:\"%H\""
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE RADPROPS_REPO_HASH 
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 "--pretty=format:\"%cd\""
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE RADPROPS_REPO_DATE 
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
  set( RADPROPS_REPO_DATE "\"No information available\"" )
  set( RADPROPS_REPO_HASH "\"No information available\"" )
endif()

if( RadProps_ENABLE_TESTING )
  message( WARNING "RadProps_ENABLE_PREPROCESSOR was activated because RadProps_ENABLE_TESTING was on" )
  set( RadProps_ENABLE_PREPROCESSOR ON )
endif() 
if( RadProps_ENABLE_PREPROCESSOR ) 
  add_definitions( -DRadProps_ENABLE_PREPROCESSOR )
endif()



#-- boost libraries (only needed for the executables)
if( RadProps_ENABLE_TESTING OR RadProps_ENABLE_PREPROCESSOR )
  
  set( boost_comps ${boost_comps} "program_options;regex;filesystem;system" )
  set( Boost_ADDITIONAL_VERSIONS ${Boost_ADDITIONAL_VERSIONS}
      "1.50" "1.50.0" "1.51" "1.51.0" "1.52" "1.52.0" "1.53" "1.53.0"
      "1.54" "1.54.0" "1.55" "1.55.0" )
  
  if( DEFINED BOOST_ROOT )
    message( STATUS "Setting boost installation location as: " ${BOOST_ROOT} )
    set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
  endif()
  
  if( DEFINED BOOST_INCLUDEDIR AND DEFINED BOOST_LIBRARYDIR )
    message( STATUS "Setting boost include location as: " ${BOOST_INCLUDEDIR} )
    message( STATUS "Setting boost library location as: " ${BOOST_LIBRARYDIR} )
    set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
  endif()
  
  if( DEFINED BOOST_ROOT )
    set( BOOST_DIR ${Boost_DIR} )  # find_boost breaks the supplied value of Boost_DIR
    # give location hints.
    message( STATUS "Setting boost installation location as: " ${Boost_DIR} )
    set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
    set( TP_CMAKE_ARGS "-DBoost_DIR=${BOOST_DIR};" ${TP_CMAKE_ARGS} )
  endif( DEFINED BOOST_ROOT )
  
  if( DEFINED BOOST_INCLUDEDIR AND DEFINED BOOST_LIBRARYDIR )
    message( STATUS "Setting boost include location as: " ${BOOST_INCLUDEDIR} )
    message( STATUS "Setting boost library location as: " ${BOOST_LIBRARYDIR} )
    set( Boost_NO_SYSTEM_PATHS ON )  # only look in the user-specified location - nowhere else!
  endif()
  
  set(BOOST_MIN_VERSION "1.49.0" )
  
  find_package( Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS ${boost_comps} )
  
  message( STATUS "RadProps: Boost header installation found at: " ${Boost_INCLUDE_DIR} )
  message( STATUS "RadProps: Boost library installation found at: " ${Boost_LIBRARY_DIRS} )
  message( STATUS "RadProps: Boost version: " ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION} )
  
  set( TPL_LIBS ${TPL_LIBS} ${Boost_LIBRARIES} )
  set( TPL_INCLUDE ${TPL_INCLUDE} ${Boost_INCLUDE_DIR} )
  
endif()

# grab TabProps and build the linear interpolants
if( TabProps_DIR )
  message( STATUS "Looking for a TabProps installation in: " ${TabProps_DIR} )
else( TabProps_DIR )
  message( STATUS "Building TabProps internally..." )
  if( NOT GIT_FOUND )
    message( SEND_ERROR "git required to build TabProps" )
  endif()
  if( ${CMAKE_VERSION} VERSION_LESS 2.8.6 )
    message( SEND_ERROR "CMAKE Version 2.8.6 or greater required for internal build of TabProps." )
  endif()
  
  set( TP_DIR ${PROJECT_BINARY_DIR}/tabprops )
  file( MAKE_DIRECTORY ${TP_DIR}/build )
  message( "${GIT_EXECUTABLE} clone --depth 1 git://software.crsim.utah.edu/TabProps.git ${TP_DIR}/src" )
  execute_process( COMMAND ${GIT_EXECUTABLE} clone --depth 1 git://software.crsim.utah.edu/TabProps.git ${TP_DIR}/src
                   RESULT_VARIABLE result )
  # configure:
  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DTabProps_PREPROCESSOR=OFF;-DTabProps_UTILS=OFF;-DTabProps_ENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${TP_DIR} ${TP_DIR}/src
                   WORKING_DIRECTORY ${TP_DIR}/build
                   RESULT_VARIABLE result  )
  # build / install    
  execute_process( COMMAND make -j2 install
                   WORKING_DIRECTORY ${TP_DIR}/build )
  set( TabProps_DIR ${TP_DIR}/share )
  message("\n\n DONE BUILDING TABPROPS \n\n")
  message( STATUS "TabProps installation is in ${TP_DIR}" )
endif( TabProps_DIR )

find_package( Interp PATHS ${TabProps_DIR} REQUIRED )
set( TPL_LIBS interp ${TPL_LIBS} )
set( TPL_INCLUDE ${TPL_INCLUDE} ${Interp_INCLUDE_DIR} )

set( RAD_SRC
    AbsCoeffGas.cpp
    Particles.cpp
  )

set( RAD_H
    AbsCoeffGas.h
    RadiativeSpecies.h
    Particles.h
  )

include_directories( ${TPL_INCLUDE} )
add_library( radprops STATIC 
  ${RAD_SRC}
  )
target_link_libraries( radprops ${TPL_LIBS} )
install( TARGETS radprops ARCHIVE DESTINATION lib/radprops )
install( FILES ${RAD_H} DESTINATION include/radprops ) 


if( RadProps_ENABLE_PREPROCESSOR )

  add_executable( radprops_prepro PreProcessor.cpp )
  target_link_libraries( radprops_prepro radprops ${TPL_LIBS} ${Boost_LIBRARIES} )
  
  add_executable( table_query TableQuery.cpp )
  target_link_libraries( table_query radprops ${TPL_LIBS} ${Boost_LIBRARIES} )
  
  install( TARGETS radprops_prepro table_query COMPONENT Runtime DESTINATION bin )

endif( RadProps_ENABLE_PREPROCESSOR )


if( RadProps_ENABLE_TESTING )
  enable_testing()
  add_subdirectory( test )
endif()

