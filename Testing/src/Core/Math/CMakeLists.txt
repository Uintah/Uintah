# Testing class instantiations
ADD_EXECUTABLE(testMiscMath ${CMAKE_SOURCE_DIR}/Testing/src/Core/Math/testMiscMath.cc)

TARGET_LINK_LIBRARIES(testMiscMath Core_Math)

ADD_TEST(Core_Math_MiscMath_Test ${EXECUTABLE_OUTPUT_PATH}/testMiscMath)
