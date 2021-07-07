set( SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/ArchesTable.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/FluidsBasedModel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TableInterface.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/TableFactory.cc
    )

if( ENABLE_ICE )
    list( APPEND SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/AdiabaticTable.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/flameSheet_rxn.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/MaterialProperties.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/Mixing.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/NonAdiabaticTable.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/PassiveScalar.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/SimpleRxn.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/MassMomEng_src.cc
        )
    if( ENABLE_MPM )
        list( APPEND SRC
                ${CMAKE_CURRENT_SOURCE_DIR}/TestModel.cc
            )
        set( LIB_DEPS ${LIB_DEPS} CCA_Components_MPM_Materials PARENT_SCOPE )
    endif()
endif()

set( MDL_SRC ${MDL_SRC} ${SRC} PARENT_SCOPE )
