add_library( Core_Geometry
        BBox.cc
        IntVector.cc
        Plane.cc
        PolyPlane.cc
        Point.cc
        Tensor.cc
        Vector.cc
    )

target_include_directories( Core_Geometry
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )

target_link_libraries( Core_Geometry
        Core_Containers
        Core_Math
        Core_Util
    )
