ADD_EXECUTABLE(puda puda.cc asci.cc jim1.cc jim2.cc rtdata.cc tecplot.cc util.cc varsummary.cc)

TARGET_LINK_LIBRARIES(puda
  Core_Containers
  Core_Exceptions
  Core_Geometry
  Core_OS
  Core_Util
  Packages_Uintah_Core_DataArchive
  Packages_Uintah_Core_Disclosure
  Packages_Uintah_Core_Exceptions
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_ProblemSpec
  Packages_Uintah_Core_Util
)