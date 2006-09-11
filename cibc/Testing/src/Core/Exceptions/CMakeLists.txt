# Exceptions
ADD_EXECUTABLE(testFileNotFound ${CMAKE_SOURCE_DIR}/Testing/src/Core/Exceptions/testFileNotFound.cc)

TARGET_LINK_LIBRARIES(testFileNotFound Core_Exceptions)

ADD_TEST(Core_Exceptions_FileNotFound_Test ${EXECUTABLE_OUTPUT_PATH}/testFileNotFound)