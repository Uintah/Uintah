#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

# Tests to exercise the classes in Core/Datatypes
##################################################################

IF(ITK_FOUND)
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
ENDIF(ITK_FOUND)

