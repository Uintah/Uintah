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



# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

INCLUDES += $(INSIGHT_INCLUDE)

SRCDIR   := Packages/Insight/Dataflow/Modules/Filters

CATEGORY := Filters

SRCS += ${SRCDIR}/BinaryDilateImageFilter.cc \
	${SRCDIR}/BinaryErodeImageFilter.cc \
	${SRCDIR}/BinaryThresholdImageFilter.cc \
	${SRCDIR}/CannySegmentationLevelSetImageFilter.cc \
	${SRCDIR}/ConfidenceConnectedImageFilter.cc \
	${SRCDIR}/ConnectedThresholdImageFilter.cc \
	${SRCDIR}/CurvatureAnisotropicDiffusionImageFilter.cc \
	${SRCDIR}/CurvatureFlowImageFilter.cc \
	${SRCDIR}/DiscreteGaussianImageFilter.cc \
	${SRCDIR}/ExtractImageFilter.cc \
	${SRCDIR}/GeodesicActiveContourLevelSetImageFilter.cc \
	${SRCDIR}/GradientAnisotropicDiffusionImageFilter.cc \
	${SRCDIR}/GradientMagnitudeImageFilter.cc \
	${SRCDIR}/GradientRecursiveGaussianImageFilter.cc \
	${SRCDIR}/GrayscaleDilateImageFilter.cc \
	${SRCDIR}/GrayscaleErodeImageFilter.cc \
	${SRCDIR}/IsolatedConnectedImageFilter.cc \
	${SRCDIR}/MeanImageFilter.cc \
	${SRCDIR}/NeighborhoodConnectedImageFilter.cc \
	${SRCDIR}/ReflectImageFilter.cc \
	${SRCDIR}/RescaleIntensityImageFilter.cc \
	${SRCDIR}/ThresholdSegmentationLevelSetImageFilter.cc \
	${SRCDIR}/UnaryFunctorImageFilter.cc \
	${SRCDIR}/VectorConfidenceConnectedImageFilter.cc \
	${SRCDIR}/VectorCurvatureAnisotropicDiffusionImageFilter.cc \
	${SRCDIR}/VectorGradientAnisotropicDiffusionImageFilter.cc \
	${SRCDIR}/VectorIndexSelectionCastImageFilter.cc \
	${SRCDIR}/WatershedImageFilter.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes \
	Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Dataflow/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry Core/GeomInterface 


LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(INSIGHT_LIBRARY) $(BLAS_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


