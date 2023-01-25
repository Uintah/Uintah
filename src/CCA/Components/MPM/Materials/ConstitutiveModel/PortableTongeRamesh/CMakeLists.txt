add_subdirectory( TongeRameshPTRCalcs )

set( MPM_SRC ${MPM_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/TongeRameshPTR.cc
        PARENT_SCOPE
    )
