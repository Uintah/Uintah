add_library( Core_GeometryPiece
        BoxGeometryPiece.cc
        ConeGeometryPiece.cc
        ConvexPolyhedronGeometryPiece.cc
        CylinderGeometryPiece.cc
        CylinderShellPiece.cc
        DifferenceGeometryPiece.cc
        EllipsoidGeometryPiece.cc
        FileGeometryPiece.cc
        #GUVSphereShellPiece.cc
        GeometryObject.cc
        GeometryPiece.cc
        GeometryPieceFactory.cc
        IntersectionGeometryPiece.cc
        LineSegGeometryPiece.cc
        NaaBoxGeometryPiece.cc
        NullGeometryPiece.cc
        PlaneShellPiece.cc
        ShellGeometryFactory.cc
        ShellGeometryPiece.cc
        SmoothCylGeomPiece.cc
        SmoothGeomPiece.cc
        SphereGeometryPiece.cc
        SphereMembraneGeometryPiece.cc
        SphereShellPiece.cc
        TorusGeometryPiece.cc
        TriGeometryPiece.cc
        UniformGrid.cc
        UnionGeometryPiece.cc
    )

target_link_libraries( Core_GeometryPiece
        Core_Exceptions
        Core_Geometry
        Core_Grid
        Core_Malloc
        Core_Math
        Core_Parallel
        Core_ProblemSpec
        Core_Util
    )

target_include_directories( Core_GeometryPiece
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )