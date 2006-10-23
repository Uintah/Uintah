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


# Makefile fragment for this subdirectory

SRCDIR := StandAlone/Apps/Painter

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else

PSELIBS := Core/Algorithms/Fields \
           Core/Algorithms/Visualization \
           Core/Basis \
           Core/Bundle \
	   Core/Containers \
	   Core/Datatypes \
           Core/Events \
	   Core/Exceptions \
           Core/Geom \
	   Core/Geometry \
           Dataflow/GuiInterface \
	   Core/Init \
           Core/Math \
	   Core/Persistent \
           Core/Skinner \
           Core/Thread \
	   Core/Util \
           Core/Volume \
           Core/XMLUtil
endif

LIBS := $(LAPACK_LIBRARY) \
        $(XML2_LIBRARY) \
        $(M_LIBRARY) \
        $(GL_LIBRARY) \
        $(TEEM_LIBRARY) \
        $(SOCKET_LIBRARY)

ifeq ($(HAVE_INSIGHT), yes)   
        LIBS += $(INSIGHT_LIBRARY) $(GDCM_LIBRARY)
endif


PROGRAM := $(SRCDIR)/painter

SRCS := $(SRCDIR)/AutoviewTool.cc \
	$(SRCDIR)/BrushTool.cc \
	$(SRCDIR)/CLUTLevelsTool.cc \
	$(SRCDIR)/CropTool.cc \
	$(SRCDIR)/FloodfillTool.cc \
        $(SRCDIR)/ITKConfidenceConnectedImageFilterTool.cc \
	$(SRCDIR)/ITKThresholdSegmentationLevelSetImageFilterTool.cc \
	$(SRCDIR)/KeyToolSelectorTool.cc \
	$(SRCDIR)/LayerButton.cc \
	$(SRCDIR)/LayerMergeTool.cc \
	$(SRCDIR)/NrrdToITK.cc \
	$(SRCDIR)/NrrdVolume.cc \
	$(SRCDIR)/Painter.cc \
        $(SRCDIR)/PainterSignalTargets.cc \
	$(SRCDIR)/PanTool.cc \
	$(SRCDIR)/ProbeTool.cc \
	$(SRCDIR)/PointerToolSelectorTool.cc \
	$(SRCDIR)/StatisticsTool.cc \
	$(SRCDIR)/SliceWindow.cc \
	$(SRCDIR)/SessionReader.cc \
	$(SRCDIR)/SessionWriter.cc \
	$(SRCDIR)/VolumeSlice.cc \
	$(SRCDIR)/ZoomTool.cc \
        $(SRCDIR)/main.cc \

include $(SCIRUN_SCRIPTS)/program.mk
