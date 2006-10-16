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

# CMakeLists.txt for Packages/ModelCreation/Dataflow/Modules/FieldsData

SET(Packages_ModelCreation_Dataflow_Modules_FieldsData_SRCS
  ComputeFieldData.cc
  ComputeFieldsData.cc
  CreateFieldData.cc
  DistanceToBoundary.cc
  DistanceToField.cc
  FieldDataElemToNode.cc
  FieldDataNodeToElem.cc
  IsInsideField.cc
  SelectAndSetFieldData.cc
  SelectAndSetFieldsData.cc
  SelectByFieldData.cc
  SelectByFieldsData.cc
  SignedDistanceToField.cc
  MappingMatrixToField.cc
  GetFieldData.cc
  SetFieldData.cc
  IndicesToData.cc
  NodalMapping.cc
  ModalMapping.cc
  CurrentDensityMapping.cc
  IsInsideFields.cc
  GradientModalMapping.cc
  FieldDataMeasure.cc
)

ADD_LIBRARY(Packages_ModelCreation_Dataflow_Modules_FieldsData ${Packages_ModelCreation_Dataflow_Modules_FieldsData_SRCS})

TARGET_LINK_LIBRARIES(Packages_ModelCreation_Dataflow_Modules_FieldsData
  Core_Datatypes
  Dataflow_Network
  Core_Persistent
  Core_Containers
  Core_Util
  Core_Exceptions
  Core_Thread
  Dataflow_GuiInterface
  Core_Geom
  Core_Datatypes
  Core_Geometry
  Core_GeomInterface
  Dataflow_TkExtensions
  Packages_ModelCreation_Core_Datatypes
  Packages_ModelCreation_Core_Fields
  Core_Algorithms_Converter
  Core_Algorithms_Fields
  Core_Algorithms_ArrayMath
  ${TK_LIBRARY}
  ${GL_LIBRARY}
  ${M_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_ModelCreation_Dataflow_Modules_FieldsData)
ENDIF(NOT BUILD_SHARED_LIBS)
