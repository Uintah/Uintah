#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
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

# CMakeLists.txt for Dataflow/Modules/Math

SET(Dataflow_Modules_Math_SRCS
  AppendMatrix.cc
  BuildNoise.cc
  BuildTransform.cc
  CastMatrix.cc
  ChooseMatrix.cc
  ErrorMetric.cc
  LinAlgBinary.cc
  LinAlgUnary.cc
  LinearAlgebra.cc
  MatrixInfo.cc
  MatrixSelectVector.cc
  MinNormLeastSq.cc
  SolveMatrix.cc
  Submatrix.cc
  MaskVectorToMappingMatrix.cc
  MappingMatrixToMaskVector.cc
  CreateMatrix.cc
)

ADD_LIBRARY(Dataflow_Modules_Math ${Dataflow_Modules_Math_SRCS})

TARGET_LINK_LIBRARIES(Dataflow_Modules_Math
  Core_Datatypes
  Core_Exceptions
  Core_Geom
  Core_GeomInterface
  Core_Geometry
  Core_Containers
  Dataflow_GuiInterface
  Core_Math
  Core_Persistent
  Core_Thread
  Dataflow_TkExtensions
  Core_Util
  Dataflow_Network
  Dataflow_Widgets
  ${TK_LIBRARY}
  ${GL_LIBRARY}
  ${M_LIBRARY}
  ${PETSC_UNI_LIBRARY}
  ${TEEM_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Dataflow_Modules_Math)
ENDIF(NOT BUILD_SHARED_LIBS)
