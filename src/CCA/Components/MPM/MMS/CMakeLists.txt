add_library( CCA_Components_MPM_MMS MMS.cc )
target_link_libraries( CCA_Components_MPM_MMS CCA_Components_MPM_Core CCA_Ports )
target_include_directories( CCA_Components_MPM_MMS
    PUBLIC
        ${CMAKE_BINARY_DIR}/include ${CMAKE_SOURCE_DIR}
    )