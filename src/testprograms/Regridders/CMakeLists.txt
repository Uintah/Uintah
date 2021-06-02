add_executable( benchmark benchmark.cc )
target_link_libraries( benchmark
        Core_Geometry
        Core_Grid
#        Core_Grid_Variables
        Core_Util
        UintahCore
    )

add_executable( outputpatches outputpatches.cc )
target_link_libraries( outputpatches Core_Grid Core_Geometry )

add_executable( patchquality patchquality.cc )
target_link_libraries( patchquality Core_Grid Core_Geometry )