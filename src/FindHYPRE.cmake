include(FindPackageHandleStandardArgs)

set( HYPRE_FOUND false )

find_path( HYPRE_INCLUDE_DIR
    NAMES HYPRE.h
    PATHS
        ENV HYPRE_DIR
        $ENV{HOME}/hypre
        ${HYPRE_DIR}/include
    DOC
        "Hypre root Directory"
    )

find_library(
        HYPRE_LIBRARY
    NAMES libHYPRE.a
    PATHS
        ENV HYPRE_DIR
        $ENV{HOME}/hypre
        ${HYPRE_DIR}/lib
    )

if( NOT (${HYPRE_LIBRARY} STREQUAL "HYPRE_LIBRARY-NOTFOUND") )
    ## Version information
    file( READ ${HYPRE_INCLUDE_DIR}/HYPRE_config.h HYPRE_HEADER )
    string( REGEX MATCH
            "HYPRE_RELEASE_VERSION \"(([0-9])+(\.){0,1})+\""
            HYPRE_VERSION
            ${HYPRE_HEADER}
        )
    message( WARNING "Not yet ready for hyper version parsing  ${HYPRE_VERSION}" )

    find_package_handle_standard_args( HYPRE
        FOUND_VAR HYPRE_FOUND
        REQUIRED_VARS
            HYPRE_LIBRARY
            HYPRE_INCLUDE_DIR
        VERSION_VAR
            HYPRE_VERSION
        )

    if( HYPRE_FOUND)
        set( HYPRE_LIBRARIES ${HYPRE_LIBRARY} )
        set( HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR} )
    endif()

    if( HYPRE_FOUND )
        if( NOT TARGET HYPRE::HYPRE )
            add_library( HYPRE::HYPRE UNKNOWN IMPORTED )
        endif()
        set_target_properties( HYPRE::HYPRE PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIR}"
            )
    endif()

    mark_as_advanced(
        HYPRE_INCLUDE_DIR
        HYPRE_LIBRARY
    )
endif()
