add_subdirectory( compare_mms )
add_subdirectory( dumpfields )
add_subdirectory( extractors )
add_subdirectory( fsspeed )
add_subdirectory( graphview )
add_subdirectory( mpi_test )
add_subdirectory( pfs )
add_subdirectory( puda )

add_executable( compute_Lnorm_udas
        compute_Lnorm_udas.cc
    )

target_link_libraries( compute_Lnorm_udas
        Core_DataArchive
        Core_Geometry
        Core_Grid
        Core_Math
        Core_OS
        Core_Parallel
    )
