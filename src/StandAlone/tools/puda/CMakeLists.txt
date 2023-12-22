add_executable( puda
        AA_MMS.cc
        asci.cc
        DOP.cc
        ER_MMS.cc
        GV_MMS.cc
        gridStats.cc
        ICE_momentum.cc
        jacquie.cc
        jim1.cc
        jim2.cc
        PIC.cc
        POL.cc
        pStressHistogram.cc
        pressure.cc
        printCellStresses.cc
        printParticleVar.cc
        puda.cc
        todd1.cc
        util.cc
        varsummary.cc
    )

target_link_libraries( puda Core_DataArchive Core_Datatypes Core_Parallel )

add_executable( grid_reader grid_reader.cc )
target_link_libraries( grid_reader Core_Parallel )

install( TARGETS puda grid_reader
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    EXPORT uintah_exe_targets
)
