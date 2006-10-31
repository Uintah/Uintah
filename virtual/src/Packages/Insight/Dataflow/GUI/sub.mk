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

SRCDIR := Packages/Insight/Dataflow/GUI

SRCS := \
	$(SRCDIR)/BinaryDilateImageFilter.tcl\
	$(SRCDIR)/BinaryErodeImageFilter.tcl\
	$(SRCDIR)/BuildSeedVolume.tcl \
	$(SRCDIR)/CannySegmentationLevelSetImageFilter.tcl \
	$(SRCDIR)/ChooseImage.tcl\
	$(SRCDIR)/ColorImageReaderFloat2D.tcl\
	$(SRCDIR)/ColorImageReaderFloat3D.tcl\
	$(SRCDIR)/ConfidenceConnectedImageFilter.tcl \
	$(SRCDIR)/ConnectedThresholdImageFilter.tcl \
	$(SRCDIR)/CurvatureAnisotropicDiffusionImageFilter.tcl \
	$(SRCDIR)/CurvatureFlowImageFilter.tcl \
	$(SRCDIR)/DiscreteGaussianImageFilter.tcl \
	$(SRCDIR)/ExtractImageFilter.tcl\
	$(SRCDIR)/GeodesicActiveContourLevelSetImageFilter.tcl \
	$(SRCDIR)/GradientAnisotropicDiffusionImageFilter.tcl \
	$(SRCDIR)/GradientMagnitudeImageFilter.tcl \
	$(SRCDIR)/GradientRecursiveGaussianImageFilter.tcl \
	$(SRCDIR)/GrayscaleDilateImageFilter.tcl\
	$(SRCDIR)/GrayscaleErodeImageFilter.tcl\
	$(SRCDIR)/ImageFileWriter.tcl \
	$(SRCDIR)/ImageReaderFloat2D.tcl \
	$(SRCDIR)/ImageReaderFloat3D.tcl \
	$(SRCDIR)/ImageToField.tcl \
	$(SRCDIR)/IsolatedConnectedImageFilter.tcl \
	$(SRCDIR)/MeanImageFilter.tcl\
	$(SRCDIR)/NeighborhoodConnectedImageFilter.tcl \
	$(SRCDIR)/ReflectImageFilter.tcl \
	$(SRCDIR)/RescaleIntensityImageFilter.tcl \
	$(SRCDIR)/SliceReader.tcl \
	$(SRCDIR)/ThresholdSegmentationLevelSetImageFilter.tcl \
	$(SRCDIR)/UnaryFunctorImageFilter.tcl \
	$(SRCDIR)/VectorConfidenceConnectedImageFilter.tcl \
	$(SRCDIR)/VectorCurvatureAnisotropicDiffusionImageFilter.tcl \
	$(SRCDIR)/VectorGradientAnisotropicDiffusionImageFilter.tcl \
	$(SRCDIR)/VectorIndexSelectionCastImageFilter.tcl \
	$(SRCDIR)/WatershedImageFilter.tcl \
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk



