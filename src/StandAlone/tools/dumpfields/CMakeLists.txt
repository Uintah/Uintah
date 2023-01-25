add_executable( dumpfields
        dumpfields.cc

        Args.cc
        EnsightDumper.cc
        FieldDiags.cc
        FieldDumper.cc
        FieldSelection.cc
        HistogramDumper.cc
        InfoDumper.cc
        ScalarDiags.cc
        TensorDiags.cc
        TextDumper.cc
        utils.cc
        VectorDiags.cc
    )

target_link_libraries( dumpfields
        Core_DataArchive
        Core_Grid
        Core_OS
    )