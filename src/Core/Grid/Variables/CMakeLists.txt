add_library( Core_Grid_Variables
        CellIterator.cc
        ComputeSet.cc
        ComputeSet_special.cc
        DifferenceIterator.cc
        GridIterator.cc
        GridSurfaceIterator.cc
        GridVariableBase.cc
        Iterator.cc
        ListOfCellsIterator.cc
        LocallyComputedPatchVarMap.cc
        NodeIterator.cc
        PSPatchMatlGhost.cc
        PSPatchMatlGhostRange.cc
        ParticleSubset.cc
        ParticleVariableBase.cc
        ParticleVariable_special.cc
        PerPatchBase.cc
        ReductionVariableBase.cc
        ReductionVariable_special.cc
        SoleVariableBase.cc
        StaticInstantiate.cc
        Stencil4.cc
        Stencil7.cc
        UnionIterator.cc
        Utils.cc
        ugc_templates.cc
        VarLabel.cc
        Variable.cc
    )

target_link_libraries( Core_Grid_Variables
        Core_Geometry
        Core_Exceptions
        Core_Util
        Core_Containers
        Core_Parallel
        Core_ProblemSpec
        Core_Exceptions
        Core_Math
        Core_Disclosure
        ${ZLIB_LIBRARIES}
        MPI::MPI_CXX
        )

# jcs PIDX...?

target_include_directories( Core_Grid_Variables
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
    )