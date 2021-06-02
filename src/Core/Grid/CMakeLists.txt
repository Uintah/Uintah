#add_subdirectory( Variables )
add_subdirectory( PatchBVH )

add_library( Core_Grid
        AMR.cc
        AMR_CoarsenRefine.cc
        AxiGIMPInterpolator.cc
        AxiLinearInterpolator.cc
        BSplineInterpolator.cc
        Box.cc
        DbgOutput.cc
        GIMPInterpolator.cc
        Ghost.cc
        Grid.cc
        Level.cc
        LinearInterpolator.cc
        Material.cc
        MaterialManager.cc
        Patch.cc
        PatchRangeTree.cc
        Region.cc
        SimpleMaterial.cc
        TOBSplineInterpolator.cc
        Task.cc
        UnknownVariable.cc
        axiCpdiInterpolator.cc
        axiCptiInterpolator.cc
        cpdiInterpolator.cc
        cptiInterpolator.cc
        fastCpdiInterpolator.cc

        BoundaryConditions/AnnulusBCData.cc
        BoundaryConditions/BCData.cc
        BoundaryConditions/BCDataArray.cc
        BoundaryConditions/BCGeomBase.cc
        BoundaryConditions/BCUtils.cc
        BoundaryConditions/BoundCondFactory.cc
        BoundaryConditions/BoundCondReader.cc
        BoundaryConditions/CircleBCData.cc
        BoundaryConditions/DifferenceBCData.cc
        BoundaryConditions/EllipseBCData.cc
        BoundaryConditions/RectangleBCData.cc
        BoundaryConditions/RectangulusBCData.cc
        BoundaryConditions/SideBCData.cc
        BoundaryConditions/UnionBCData.cc

        Variables/CellIterator.cc
        Variables/ComputeSet.cc
        Variables/ComputeSet_special.cc
        Variables/DifferenceIterator.cc
        Variables/GridIterator.cc
        Variables/GridSurfaceIterator.cc
        Variables/GridVariableBase.cc
        Variables/Iterator.cc
        Variables/ListOfCellsIterator.cc
        Variables/LocallyComputedPatchVarMap.cc
        Variables/NodeIterator.cc
        Variables/PSPatchMatlGhost.cc
        Variables/PSPatchMatlGhostRange.cc
        Variables/ParticleSubset.cc
        Variables/ParticleVariableBase.cc
        Variables/ParticleVariable_special.cc
        Variables/PerPatchBase.cc
        Variables/ReductionVariableBase.cc
        Variables/ReductionVariable_special.cc
        Variables/SoleVariableBase.cc
        Variables/StaticInstantiate.cc
        Variables/Stencil4.cc
        Variables/Stencil7.cc
        Variables/UnionIterator.cc
        Variables/Utils.cc
        Variables/ugc_templates.cc
        Variables/VarLabel.cc
        Variables/Variable.cc

        PatchBVH/PatchBVH.cc
        PatchBVH/PatchBVHBase.cc
        PatchBVH/PatchBVHLeaf.cc
        PatchBVH/PatchBVHNode.cc
    )

target_link_libraries( Core_Grid
        Core_Containers
        Core_Disclosure
        Core_Exceptions
        Core_Geometry
        Core_Malloc
        Core_Math
        Core_Parallel
        Core_ProblemSpec
        Core_Util
    )

# jcs what about PIDX here?

target_include_directories( Core_Grid
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
    )