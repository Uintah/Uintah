set( MPM_SRC MPMMaterial.cc )  # collect files in subfolders

add_subdirectory( ConstitutiveModel )
add_subdirectory( Contact )
add_subdirectory( Diffusion )
add_subdirectory( ParticleCreator )

add_library( CCA_Components_MPM_Materials ${MPM_SRC} )

target_link_libraries( CCA_Components_MPM_Materials
        CCA_Components_MPM_MMS
        CCA_Components_MPM_PhysicalBC
        Core_Exceptions
        Core_ProblemSpec
        Core_Util
    )

target_include_directories( CCA_Components_MPM_Materials
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
    )
