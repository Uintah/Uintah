ADD_EXECUTABLE(pfs pfs.cc)
ADD_EXECUTABLE(pfs2 pfs2.cc)

SET(PFS_LIBS
  Core_Exceptions
  Packages_Uintah_Core_Exceptions
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_Parallel
  Packages_Uintah_Core_ProblemSpec
  Packages_Uintah_Core_Util
  Packages_Uintah_CCA_Components_ProblemSpecification
)

TARGET_LINK_LIBRARIES(pfs ${PFS_LIBS})
TARGET_LINK_LIBRARIES(pfs2 ${PFS_LIBS})