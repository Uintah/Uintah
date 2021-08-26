set( OS_SRC ProcessInfo.cc )

if( HAVE_BOOST )
    set( OS_SRC ${OS_SRC} Dir_boost.cc )
    set( LINK_DEPS ${Boost_LIBRARIES} )
    set( INC_DEPS ${Boost_INCLUDE_DIRS} )
else()
    set( OS_SRC ${OS_SRC} Dir.cc )
    set( LINK_DEPS "" )
    set( INC_DEPS "" )
endif()

add_library( Core_OS ${OS_SRC} )

target_link_libraries( Core_OS
        Core_Exceptions
        Core_Parallel
        Core_Util
        ${LINK_DEPS}
    )

target_include_directories( Core_OS
        PUBLIC
            ${CMAKE_SOURCE_DIR}
        PRIVATE
            ${INC_DEPS}
    )
