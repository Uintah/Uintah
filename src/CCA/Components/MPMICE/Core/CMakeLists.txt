add_library( CCA_Components_MPMICE_Core MPMICELabel.cc )
target_link_libraries( CCA_Components_MPMICE_Core Core_Grid )
target_include_directories( CCA_Components_MPMICE_Core PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} )