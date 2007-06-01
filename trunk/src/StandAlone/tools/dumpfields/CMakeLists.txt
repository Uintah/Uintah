ADD_EXECUTABLE(dumpfields dumpfields.cc
  utils.cc
  Args.cc
  FieldSelection.cc
  FieldDiags.cc
  ScalarDiags.cc
  VectorDiags.cc
  TensorDiags.cc
  FieldDumper.cc
  TextDumper.cc
  EnsightDumper.cc
  InfoDumper.cc
  HistogramDumper.cc
)

TARGET_LINK_LIBRARIES(dumpfields
   Packages_Uintah_Core_DataArchive   
   Core_OS
)