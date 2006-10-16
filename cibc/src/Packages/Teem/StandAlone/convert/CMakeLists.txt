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

# CMakeLists.txt for Packages/Teem/StandAlone/convert

SET(Packages_Teem_StandAlone_convert_SRCS
  MakeTensorsIsotropic.cc
)

ADD_LIBRARY(Packages_Teem_StandAlone_convert ${Packages_Teem_StandAlone_convert_SRCS})

TARGET_LINK_LIBRARIES(Packages_Teem_StandAlone_convert
  Core_Containers
  Core_Persistent
  Core_Exceptions
  Core_Math
  Core_Thread
  Core_Geometry
  Core_Util
  ${TEEM_LIBRARY}
  ${LAPACK_LIBRARY}
  ${XML2_LIBRARY}
  ${M_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Teem_StandAlone_convert)
ENDIF(NOT BUILD_SHARED_LIBS)
