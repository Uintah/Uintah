add_subdirectory( Containers )
add_subdirectory( DataArchive )
add_subdirectory( Datatypes )
add_subdirectory( Disclosure )
add_subdirectory( Exceptions )
add_subdirectory( Geometry )
add_subdirectory( GeometryPiece )
add_subdirectory( Grid )
add_subdirectory( IO )
#add_subdirectory( Lockfree )
add_subdirectory( Malloc )
add_subdirectory( Math )
add_subdirectory( OS )
add_subdirectory( Parallel )
add_subdirectory( ProblemSpec )
add_subdirectory( Util )

# This is a convenience library that will grab all of the Core libraries
add_library( UintahCore INTERFACE )

target_link_libraries( UintahCore INTERFACE
        Core_Containers
        Core_DataArchive
        Core_Datatypes
        Core_Disclosure
        Core_Exceptions
        Core_Geometry
        Core_GeometryPiece
        Core_Grid
#        Core_Grid_BoundaryConditions
#        Core_Grid_PatchBVH
#        Core_Grid_Variables
        Core_IO
        Core_Malloc
        Core_Math
        Core_OS
        Core_Parallel
        Core_ProblemSpec
        Core_Util
    )
