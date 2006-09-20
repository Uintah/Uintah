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

# CMakeLists.txt for Packages/Insight/Dataflow/Modules/Filters

SET(Packages_Insight_Dataflow_Modules_Filters_SRCS
  ${SRCDIR}/BinaryDilateImageFilter.cc
  ${SRCDIR}/BinaryErodeImageFilter.cc
  ${SRCDIR}/BinaryThresholdImageFilter.cc
  ${SRCDIR}/CannySegmentationLevelSetImageFilter.cc
  ${SRCDIR}/ConfidenceConnectedImageFilter.cc
  ${SRCDIR}/ConnectedThresholdImageFilter.cc
  ${SRCDIR}/CurvatureAnisotropicDiffusionImageFilter.cc
  ${SRCDIR}/CurvatureFlowImageFilter.cc
  ${SRCDIR}/DiscreteGaussianImageFilter.cc
  ${SRCDIR}/ExtractImageFilter.cc
  ${SRCDIR}/GeodesicActiveContourLevelSetImageFilter.cc
  ${SRCDIR}/GradientAnisotropicDiffusionImageFilter.cc
  ${SRCDIR}/GradientMagnitudeImageFilter.cc
  ${SRCDIR}/GradientRecursiveGaussianImageFilter.cc
  ${SRCDIR}/GrayscaleDilateImageFilter.cc
  ${SRCDIR}/GrayscaleErodeImageFilter.cc
  ${SRCDIR}/IsolatedConnectedImageFilter.cc
  ${SRCDIR}/MeanImageFilter.cc
  ${SRCDIR}/NeighborhoodConnectedImageFilter.cc
  ${SRCDIR}/ReflectImageFilter.cc
  ${SRCDIR}/RescaleIntensityImageFilter.cc
  ${SRCDIR}/ThresholdSegmentationLevelSetImageFilter.cc
  ${SRCDIR}/UnaryFunctorImageFilter.cc
  ${SRCDIR}/VectorConfidenceConnectedImageFilter.cc
  ${SRCDIR}/VectorCurvatureAnisotropicDiffusionImageFilter.cc
  ${SRCDIR}/VectorGradientAnisotropicDiffusionImageFilter.cc
  ${SRCDIR}/VectorIndexSelectionCastImageFilter.cc
  ${SRCDIR}/WatershedImageFilter.cc
)

ADD_LIBRARY(Packages_Insight_Dataflow_Modules_Filters ${Packages_Insight_Dataflow_Modules_Filters_SRCS})

TARGET_LINK_LIBRARIES(Packages_Insight_Dataflow_Modules_Filters
  Core_Datatypes
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
  ${TK_LIBRARY}
  ${GL_LIBRARY}
  ${M_LIBRARY}
  ${INSIGHT_LIBRARY}
  ${BLAS_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Insight_Dataflow_Modules_Filters)
ENDIF(NOT BUILD_SHARED_LIBS)
