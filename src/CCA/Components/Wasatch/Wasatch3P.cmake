# Look for required libraries:
#   SpatialOps, ExprLib, TabProps, RadProps

####################################################################################
# GIT Hash Tags for the various libraries - used if W3P is automatically built here.
set( SPATIAL_OPS_TAG    bd7a2bac76e42e2cb528b02e2cf1291d05c71926 )
set( TAB_PROPS_TAG      eca0dc5f26b20d4f6b1cebe51b1ae97e11871e83 )
set( RAD_PROPS_TAG      bd95421430fc266ee88d0480069f7d20be1414f6 )
set( EXPR_LIB_TAG       d2950c77edb22e5f3669ddaaf2d2ed29ceb24d2c )
set( NSCBC_TAG          2e355b392f750f99c29b52baa7d64245bcdd0df1 )
set( POKITT_TAG         fdfdaa225a950384da1234c1be4e03e6dc2604cc )
####################################################################################


#--------------------------------------
# find RadProps, TabProps, NSCBC, PoKiTT, ExprLib, and SpatialOps - or build them if requested.

if( WASATCH_BUILD_W3P_LIBS )

    if( NOT GIT_FOUND )
        message( SEND_ERROR "git was not found so upstream libraries cannot be automatically built" )
    endif()

    set( W3P_DIR ${PROJECT_BINARY_DIR}/w3p )
    message( STATUS "Building W3P Libraries" )

    include( FetchContent )
    set( FETCHCONTENT_BASE_DIR ${W3P_DIR} )

    #--- SpatialOps
    FetchContent_Declare(
            spatialops_builder
            GIT_REPOSITORY https://gitlab.multiscale.utah.edu/common/SpatialOps.git
            GIT_TAG ${SPATIAL_OPS_TAG}
            TIMEOUT 60
        )
    FetchContent_GetProperties( spatialops_builder )
    if( NOT spatialops_builder_POPULATED )
        FetchContent_Populate( spatialops_builder )
        message( STATUS "Building SpatialOps in ${spatialops_builder_BINARY_DIR}  (patience...)" )
        execute_process( COMMAND ${CMAKE_COMMAND} ${spatialops_builder_SOURCE_DIR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_INSTALL_PREFIX=${W3P_DIR}
                -DENABLE_THREADS=OFF
                -DENABLE_CUDA=OFF
                -DENABLE_TESTS=OFF
                -DENABLE_EXAMPLES=OFF
                -DBOOST_INCLUDEDIR=${Boost_INCLUDE_DIRS}
                -DBOOST_LIBRARYDIR=${Boost_LIBRARY_DIRS}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                WORKING_DIRECTORY   ${spatialops_builder_BINARY_DIR}
                OUTPUT_FILE         ${spatialops_builder_BINARY_DIR}/build_output.log
                ERROR_FILE          ${spatialops_builder_BINARY_DIR}/build_errors.log
                RESULT_VARIABLE     result
            )
        execute_process( COMMAND ${CMAKE_COMMAND} --build ${spatialops_builder_BINARY_DIR} --target install -j8
                WORKING_DIRECTORY   ${spatialops_builder_BINARY_DIR}
                OUTPUT_FILE         ${spatialops_builder_BINARY_DIR}/build_output.log
                ERROR_FILE          ${spatialops_builder_BINARY_DIR}/build_errors.log
                RESULT_VARIABLE     result
            )
        message( STATUS "SpatialOps build complete" )
        if(result)
            message(FATAL_ERROR "Failed SpatialOps build, see build error log at:"
                    "\t${spatialops_builder_BINARY_DIR}/build_errors.log")
        endif()
    endif()

    #--- ExprLib
    FetchContent_Declare(
            exprlib_builder
            GIT_REPOSITORY https://gitlab.multiscale.utah.edu/common/ExprLib.git
            GIT_TAG ${EXPR_LIB_TAG}
            TIMEOUT 60
        )
    FetchContent_GetProperties( exprlib_builder )
    if( NOT exprlib_builder_POPULATED )
        set( SpatialOps_DIR ${W3P_DIR}/share )
        FetchContent_Populate( exprlib_builder )
        message( STATUS "Building ExprLib in ${exprlib_builder_BINARY_DIR} (patience...)" )
        execute_process( COMMAND ${CMAKE_COMMAND} ${exprlib_builder_SOURCE_DIR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_INSTALL_PREFIX=${W3P_DIR}
                -DSpatialOps_DIR=${W3P_DIR}/share
                -DBOOST_INCLUDEDIR=${Boost_INCLUDE_DIRS}
                -DBOOST_LIBRARYDIR=${Boost_LIBRARY_DIRS}
                -DENABLE_UINTAH=ON
                -DENABLE_TESTS=OFF
                -DBUILD_GUI=OFF
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                OUTPUT_FILE         ${exprlib_builder_BINARY_DIR}/build_output.log
                RESULT_VARIABLE     result
                WORKING_DIRECTORY ${exprlib_builder_BINARY_DIR}
            )
        execute_process( COMMAND make -j8 install
                WORKING_DIRECTORY   ${exprlib_builder_BINARY_DIR}
                OUTPUT_FILE         ${exprlib_builder_BINARY_DIR}/build_output.log
                RESULT_VARIABLE     result
            )
        message(STATUS "ExprLib build complete")
        if(result)
            message( FATAL_ERROR "Failed ExprLib build" )
        endif()
    endif()

    #--- TabProps
    FetchContent_Declare(
            tabprops_builder
            GIT_REPOSITORY https://gitlab.multiscale.utah.edu/common/TabProps.git
            GIT_TAG ${TAB_PROPS_TAG}
            TIMEOUT 60
        )
    FetchContent_GetProperties( tabprops_builder )
    if( NOT tabprops_builder_POPULATED )
        FetchContent_Populate( tabprops_builder )
        message( STATUS "Building TabProps in ${tabprops_builder_BINARY_DIR} (patience...)" )
        execute_process( COMMAND ${CMAKE_COMMAND} ${tabprops_builder_SOURCE_DIR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_INSTALL_PREFIX=${W3P_DIR}
                -DTabProps_PREPROCESSOR=OFF
                -DTabProps_UTILS=OFF
                -DTabProps_ENABLE_TESTING=OFF
                -DBOOST_INCLUDEDIR=${Boost_INCLUDE_DIRS}
                -DBOOST_LIBRARYDIR=${Boost_LIBRARY_DIRS}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                WORKING_DIRECTORY   ${tabprops_builder_BINARY_DIR}
                OUTPUT_FILE         ${tabprops_builder_BINARY_DIR}/build_output.log
                RESULT_VARIABLE     result
            )
        execute_process( COMMAND ${CMAKE_COMMAND} --build ${tabprops_builder_BINARY_DIR} --target install -j8
                WORKING_DIRECTORY   ${tabprops_builder_BINARY_DIR}
                OUTPUT_FILE         ${tabprops_builder_BINARY_DIR}/build_output.log
                RESULT_VARIABLE     result
            )
        message( STATUS "TabProps build complete" )
        if( result )
            message( FATAL_ERROR "Failed TabProps build" )
        endif()
    endif()


    #--- RadProps
    FetchContent_Declare(
            radprops_builder
            GIT_REPOSITORY https://gitlab.multiscale.utah.edu/common/RadProps.git
            GIT_TAG ${RAD_PROPS_TAG}
            TIMEOUT 60
    )
    FetchContent_GetProperties( radprops_builder )
    if( NOT radprops_builder_POPULATED )
        FetchContent_Populate( radprops_builder )
        message( STATUS "Building RadProps in ${radprops_builder_BINARY_DIR} (patience...)" )
        execute_process( COMMAND ${CMAKE_COMMAND} ${radprops_builder_SOURCE_DIR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_INSTALL_PREFIX=${W3P_DIR}
                -DRadProps_ENABLE_PREPROCESSOR=OFF
                -DRadProps_ENABLE_TESTING=OFF
                -DTabProps_DIR=${W3P_DIR}/share
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                WORKING_DIRECTORY   ${radprops_builder_BINARY_DIR}
                OUTPUT_FILE         ${radprops_builder_BINARY_DIR}/build_output.log
                RESULT_VARIABLE     result
            )
        execute_process( COMMAND ${CMAKE_COMMAND} --build ${radprops_builder_BINARY_DIR} --target install -j8
                WORKING_DIRECTORY   ${radprops_builder_BINARY_DIR}
                OUTPUT_FILE         ${radprops_builder_BINARY_DIR}/build_output.log
                RESULT_VARIABLE     result
            )
        message( STATUS "RadProps build complete" )
        if( result )
            message( FATAL_ERROR "Failed RdProps build" )
        endif()
    endif()

    #--- NSCBC
    FetchContent_Declare(
            nscbc_builder
            GIT_REPOSITORY https://gitlab.multiscale.utah.edu/common/NSCBC.git
            GIT_TAG ${NSCBC_TAG}
            TIMEOUT 60
    )
    FetchContent_GetProperties( nscbc_builder )
    if( NOT nscbc_builder_POPULATED )
        FetchContent_Populate( nscbc_builder )
        message( STATUS "Building NSCBC in ${nscbc_builder_BINARY_DIR} (patience...)" )
        execute_process( COMMAND ${CMAKE_COMMAND} ${nscbc_builder_SOURCE_DIR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_INSTALL_PREFIX=${W3P_DIR}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                WORKING_DIRECTORY   ${nscbc_builder_BINARY_DIR}
                OUTPUT_FILE         ${nscbc_builder_BINARY_DIR}/build_output.log
                RESULT_VARIABLE     result
                )
        execute_process( COMMAND ${CMAKE_COMMAND} --build ${nscbc_builder_BINARY_DIR} --target install -j8
                WORKING_DIRECTORY   ${nscbc_builder_BINARY_DIR}
                OUTPUT_FILE         ${nscbc_builder_BINARY_DIR}/build_output.log
                RESULT_VARIABLE     result
                )
        message( STATUS "NSCBC build complete" )
        if( result )
            message( FATAL_ERROR "Failed NSCBC build" )
        endif()
    endif()


    if( WASATCH_ENABLE_POKITT )

        set( Cantera_DIR "" CACHE PATH "Path to Cantera installation" )
        set( YAML_DIR "" CACHE PATH "Path to yaml-cpp installation" )

        # --- PoKiTT
        FetchContent_Declare(
            pokitt_builder
            GIT_REPOSITORY https://gitlab.multiscale.utah.edu/common/PoKiTT.git
            GIT_TAG ${POKITT_TAG}
            TIMEOUT 60
        )
        FetchContent_GetProperties( pokitt_builder )
        if( NOT pokitt_builder_POPULATED )
            FetchContent_Populate( pokitt_builder )
            message( STATUS "Building PoKiTT in ${pokitt_builder_BINARY_DIR} (patience...)" )

            execute_process( COMMAND ${CMAKE_COMMAND} ${pokitt_builder_SOURCE_DIR}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DCMAKE_INSTALL_PREFIX=${W3P_DIR}
                    -DBUILD_UPSTREAM_LIBS=OFF
                    -DExprLib_DIR=${W3P_DIR}/share
                    -DENABLE_TESTS=OFF
                    -DENABLE_EXAMPLES=OFF
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCantera_DIR=${Cantera_DIR}
                    -DYAML_DIR=${YAML_DIR}
                    WORKING_DIRECTORY   ${pokitt_builder_BINARY_DIR}
                    OUTPUT_FILE         ${pokitt_builder_BINARY_DIR}/build_output.log
                    RESULT_VARIABLE     result
                )
            execute_process( COMMAND ${CMAKE_COMMAND} --build ${pokitt_builder_BINARY_DIR} --target install -j8
                    WORKING_DIRECTORY   ${pokitt_builder_BINARY_DIR}
                    OUTPUT_FILE         ${pokitt_builder_BINARY_DIR}/build_output.log
                    RESULT_VARIABLE     result
                )
            message( STATUS "NSCBC build complete" )

            find_package( PoKiTT REQUIRED PATHS ${W3P_DIR} PATH_SUFFIXES share )
            set( W3P_LIBS pokitt )
        endif()

    else()
        find_package( ExprLib REQUIRED PATHS ${W3P_DIR} PATH_SUFFIXES share )
        set( W3P_LIBS exprlib )
    endif()

    find_package( RadProps REQUIRED PATHS ${W3P_DIR} PATH_SUFFIXES share )
    find_package( NSCBC    REQUIRED PATHS ${W3P_DIR} PATH_SUFFIXES share )
    find_package( ExprLib  REQUIRED PATHS ${W3P_DIR} PATH_SUFFIXES share )
    list( APPEND W3P_LIBS exprlib RadProps::radprops TabProps::tabprops nscbc )

else( WASATCH_BUILD_W3P_LIBS )
    set( W3P_LIBS )

    set( RadProps_DIR "" CACHE PATH "Path to installation (share) dir for RadProps" )
    find_package( RadProps REQUIRED PATHS ${RadProps_DIR} PATH_SUFFIXES share )
    find_package( NSCBC    REQUIRED PATHS ${NSCBC_DIR}    PATH_SUFFIXES share )
    list( APPEND W3P_LIBS RadProps::radprops TabProps::tabprops nscbc )

    if( WASATCH_ENABLE_POKITT )
        set( PoKiTT_DIR "" CACHE PATH "Path to installation of PoKiTT (share dir)" )
        find_package( PoKiTT REQUIRED PATHS ${PoKiTT_DIR} PATH_SUFFIXES share )
        list( APPEND W3P_LIBS pokitt )
    else()
        set( ExprLib_DIR "" CACHE PATH "Path to installation (share) dir for ExprLib" )
        set( NSCBC_DIR "" CACHE PATH "Path to installation (share) dir for NSCBC" )
        find_package( ExprLib REQUIRED PATHS ${ExprLib_DIR} PATH_SUFFIXES share )
        list( APPEND W3P_LIBS exprlib )
    endif()

endif( WASATCH_BUILD_W3P_LIBS )

message( STATUS "Wasatch Upstream Library information:"
    "\n\tRadProps:   ${RadProps_INCLUDE_DIR} "
    "\n\tTabProps:   ${TabProps_INCLUDE_DIR} "
    "\n\tSpatialOps: ${SpatialOps_INCLUDE_DIR}"
    "\n\tNSCBC:      ${NSCBC_INCLUDE_DIR}"
    "\n\tExprLib:    ${ExprLib_INCLUDE_DIR}"
    )
if( WASATCH_ENABLE_POKITT )
    message( "\tPoKiTT:     ${PoKiTT_INCLUDE_DIR}")
endif()