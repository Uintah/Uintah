add_executable( pfs  pfs.cc )
add_executable( pfs2 pfs2.cc)
add_executable( rawToUniqueGrains rawToUniqueGrains.cc )
add_executable( ImageFromGeom ImageFromGeom.cc )

target_link_libraries( pfs CCA_Components_ProblemSpecification )
target_link_libraries( pfs2 CCA_Components_ProblemSpecification Core_GeometryPiece )
target_link_libraries( rawToUniqueGrains CCA_Components_ProblemSpecification )
target_link_libraries( ImageFromGeom Core_Geometry Core_Parallel )
