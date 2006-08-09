# Tests to exercise the classes in Core/Containers
##################################################################


# ConsecuteRangeSet .h/.cc
ADD_EXECUTABLE(testConsecutiveRangeSet ${CMAKE_SOURCE_DIR}/Testing/src/Core/Containers/testConsecutiveRangeSet.cc)

TARGET_LINK_LIBRARIES(testConsecutiveRangeSet Core_Containers)

ADD_TEST(Core_Containers_ConsecutiveRangeSet_Test ${EXECUTABLE_OUTPUT_PATH}/testConsecutiveRangeSet)

# StringUtil .h/.cc
ADD_EXECUTABLE(testStringUtil ${CMAKE_SOURCE_DIR}/Testing/src/Core/Containers/testStringUtil.cc)

TARGET_LINK_LIBRARIES(testStringUtil Core_Containers)

ADD_TEST(Core_Containers_StringUtil_Test ${EXECUTABLE_OUTPUT_PATH}/testStringUtil)