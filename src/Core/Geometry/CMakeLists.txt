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
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

target_link_libraries( Core_Geometry
        Core_Containers
        Core_Math
        Core_Util
    )

install( FILES BBox.h GPUVector.h IntVector.h Plane.h Point.h PolyPlane.h Tensor.h Vector.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Core/Geometry
)

install( TARGETS Core_Geometry
    EXPORT uintah_lib_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
