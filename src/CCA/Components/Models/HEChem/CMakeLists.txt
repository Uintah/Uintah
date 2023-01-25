if( ENABLE_ICE AND ENABLE_MPM )
    set( MDL_HECHEM_SRC ${MDL_HECHEM_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/Common.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/DDT0.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/DDT1.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/IandG.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/JWLpp.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/LightTime.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/MesoBurn.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/Simple_Burn.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/Steady_Burn.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/Unsteady_Burn.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ZeroOrder.cc
            PARENT_SCOPE
        )
endif()
