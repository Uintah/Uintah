if( ENABLE_ICE AND ENABLE_MPM )
    set( MDL_SRC ${MDL_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/Arrhenius.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/AvaramiErofeevModel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ContractingCylinderModel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ContractingSphereModel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/DiffusionModel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedArrhenius.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/NthOrderModel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/PowerModel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ProutTompkinsModel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/SolidReactionModel.cc
            PARENT_SCOPE
        )
endif()
