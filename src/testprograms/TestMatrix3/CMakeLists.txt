add_library( TestMatrix3 testmatrix3.cc )
target_link_libraries( TestMatrix3
        TestSuite
        CCA_Ports
        Core_Exceptions
        Core_Geometry
        Core_Grid
        Core_Parallel
        Core_Math
    )