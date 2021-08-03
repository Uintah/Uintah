add_subdirectory( CCK )
add_subdirectory( FirstOrderArrhenius )
add_subdirectory( LangmuirHinshelwood )

set( WASATCH_SRC ${WASATCH_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/CharBase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CharData.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CharInterface.cc
        PARENT_SCOPE
    )
