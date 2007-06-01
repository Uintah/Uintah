ADD_EXECUTABLE(timeextract timeextract.cc)
ADD_EXECUTABLE(lineextract lineextract.cc)
ADD_EXECUTABLE(partextract partextract.cc)
ADD_EXECUTABLE(faceextract faceextract.cc)
ADD_EXECUTABLE(extractF extractF.cc)
ADD_EXECUTABLE(extractV extractV.cc)
ADD_EXECUTABLE(extractX extractS.cc)

SET(EXTRACTOR_LIBS
  Core_Exceptions
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_DataArchive
)

TARGET_LINK_LIBRARIES(timeextract ${EXTRACTOR_LIBS})
TARGET_LINK_LIBRARIES(lineextract ${EXTRACTOR_LIBS})
TARGET_LINK_LIBRARIES(partextract ${EXTRACTOR_LIBS})
TARGET_LINK_LIBRARIES(faceextract ${EXTRACTOR_LIBS})
TARGET_LINK_LIBRARIES(extractF ${EXTRACTOR_LIBS})
TARGET_LINK_LIBRARIES(extractV ${EXTRACTOR_LIBS})
TARGET_LINK_LIBRARIES(extractX ${EXTRACTOR_LIBS})
