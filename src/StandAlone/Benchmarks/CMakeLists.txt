add_executable( SimpleMath SimpleMath.cc )
target_link_libraries( SimpleMath
        Core_Geometry
        Core_Grid
#        Core_Grid_Variables
        Core_Util
    )
