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

# CMakeLists.txt for Packages/Insight/Dataflow/Modules/Filters

SET(Packages_Insight_Dataflow_Modules_Filters_SRCS
  BinaryDilateImageFilter.cc
  BinaryErodeImageFilter.cc
  BinaryThresholdImageFilter.cc
  CannySegmentationLevelSetImageFilter.cc
  ConfidenceConnectedImageFilter.cc
  ConnectedThresholdImageFilter.cc
  CurvatureAnisotropicDiffusionImageFilter.cc
  CurvatureFlowImageFilter.cc
  DiscreteGaussianImageFilter.cc
  ExtractImageFilter.cc
  GeodesicActiveContourLevelSetImageFilter.cc
  GradientAnisotropicDiffusionImageFilter.cc
  GradientMagnitudeImageFilter.cc
  GradientRecursiveGaussianImageFilter.cc
  GrayscaleDilateImageFilter.cc
  GrayscaleErodeImageFilter.cc
  IsolatedConnectedImageFilter.cc
  MeanImageFilter.cc
  NeighborhoodConnectedImageFilter.cc
  ReflectImageFilter.cc
  RescaleIntensityImageFilter.cc
  ThresholdSegmentationLevelSetImageFilter.cc
  UnaryFunctorImageFilter.cc
  VectorConfidenceConnectedImageFilter.cc
  VectorCurvatureAnisotropicDiffusionImageFilter.cc
  VectorGradientAnisotropicDiffusionImageFilter.cc
  VectorIndexSelectionCastImageFilter.cc
  WatershedImageFilter.cc
)

ADD_LIBRARY(Packages_Insight_Dataflow_Modules_Filters ${Packages_Insight_Dataflow_Modules_Filters_SRCS})

TARGET_LINK_LIBRARIES(Packages_Insight_Dataflow_Modules_Filters
  Core_Datatypes
  Dataflow_Network
  Core_Persistent
  Core_Containers
  Core_Util
  Core_Exceptions
  Core_Thread
  Dataflow_GuiInterface
  Core_Geom
  Core_Geometry
  Core_GeomInterface
  ITKCommon
  ITKIO
  ITKBasicFilters
  ITKAlgorithms
  ${TK_LIBRARY}
  ${GL_LIBRARY}
  ${M_LIBRARY}
  ${BLAS_LIBRARY}
)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Insight_Dataflow_Modules_Filters)
ENDIF(BUILD_SHARED_LIBS)
