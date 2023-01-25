add_library( Core_Grid_PatchBVH
        PatchBVH.cc
        PatchBVHBase.cc
        PatchBVHLeaf.cc
        PatchBVHNode.cc
    )

target_link_libraries( Core_Grid_PatchBVH
        Core_Geometry
        Core_Grid
        Core_Malloc
    )

target_include_directories( Core_Grid_PatchBVH PUBLIC ${CMAKE_SOURCE_DIR} )
