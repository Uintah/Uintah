if( WASATCH_ENABLE_POKITT )
    set( WASATCH_SRC ${WASATCH_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/TestLowMachSpeciesDensitySolver.cc
            PARENT_SCOPE
        )
endif()