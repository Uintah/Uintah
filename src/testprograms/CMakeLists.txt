add_subdirectory( CubeRootTest )
add_subdirectory( IteratorTest )
add_subdirectory( Malloc )
add_subdirectory( PatchBVH )
add_subdirectory( RegionTest )
add_subdirectory( Regridders )
add_subdirectory( SFCTest )
add_subdirectory( TestBoxGrouper )
add_subdirectory( TestConsecutiveRangeSet )
add_subdirectory( TestFastMatrix )
add_subdirectory( TestMatrix3 )
add_subdirectory( TestRangeTree )
add_subdirectory( TestSuite )

add_executable( RunTests RunTests.cc )
target_link_libraries( RunTests
        TestSuite TestMatrix3 TestConsecutiveRangeSet TestRangeTree TestBoxGrouper
        UintahCore
    )
