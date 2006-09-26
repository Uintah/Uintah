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

# CMakeLists.txt for Core/Algorithms/Visualization

SET(Core_Algorithms_Visualization_SRCS
  EdgeMC.cc
  HexMC.cc
  MarchingCubes.cc
  mcube2.cc
  Noise.cc
  NrrdTextureBuilderAlgo.cc
  PrismMC.cc
  QuadMC.cc
  RenderField.cc
  Sage.cc
  TetMC.cc
  TextureBuilderAlgo.cc
  TriMC.cc
  UHexMC.cc
)

ADD_LIBRARY(Core_Algorithms_Visualization ${Core_Algorithms_Visualization_SRCS})

TARGET_LINK_LIBRARIES(Core_Algorithms_Visualization
  Core_Basis
  Core_Containers
  Core_Datatypes
  Core_Exceptions
  Core_Geom
  Core_Geometry
  Core_Persistent
  Core_Thread
  Core_Util
  Core_Volume
  ${TEEM_LIBRARY}
  ${PNG_LIBRARY}
  ${Z_LIBRARY}
  ${GL_LIBRARY}
  ${M_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Algorithms_Visualization)
ENDIF(NOT BUILD_SHARED_LIBS)
