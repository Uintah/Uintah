# Tests to exercise the classes in Core/Datatypes
##################################################################

IF(HAVE_INSIGHT) 
  # ITKDatatype
  ADD_EXECUTABLE(testITKDatatype ${CMAKE_SOURCE_DIR}/Testing/src/Core/Datatypes/testITKDatatype.cc)

  TARGET_LINK_LIBRARIES(testITKDatatype Core_Datatypes Core_Malloc)

  ADD_TEST(Core_Datatypes_ITKDatatype_Test
    ${EXECUTABLE_OUTPUT_PATH}/testITKDatatype)

  # ITKImageField
  ADD_EXECUTABLE(testITKImageField ${CMAKE_SOURCE_DIR}/Testing/src/Core/Datatypes/testITKImageField.cc)

  TARGET_LINK_LIBRARIES(testITKImageField 
    Core_Datatypes 
    Core_Malloc)

  ADD_TEST(Core_Datatypes_ITKImageField_Test
    ${EXECUTABLE_OUTPUT_PATH}/testITKImageField)

  # ITKLatVolField
  ADD_EXECUTABLE(testITKLatVolField ${CMAKE_SOURCE_DIR}/Testing/src/Core/Datatypes/testITKLatVolField.cc)

  TARGET_LINK_LIBRARIES(testITKLatVolField 
    Core_Datatypes 
    Core_Malloc)

  ADD_TEST(Core_Datatypes_ITKLatVolField_Test
    ${EXECUTABLE_OUTPUT_PATH}/testITKLatVolField)
ENDIF(HAVE_INSIGHT) 

