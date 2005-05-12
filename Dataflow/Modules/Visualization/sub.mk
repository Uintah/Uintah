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


# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/AddLight.cc\
	$(SRCDIR)/ChooseColorMap.cc\
	$(SRCDIR)/EditColorMap.cc\
	$(SRCDIR)/EditColorMap2D.cc\
	$(SRCDIR)/GenAxes.cc\
	$(SRCDIR)/GenClock.cc\
	$(SRCDIR)/GenStandardColorMaps.cc\
	$(SRCDIR)/GenTitle.cc\
	$(SRCDIR)/Isosurface.cc\
	$(SRCDIR)/RescaleColorMap.cc\
	$(SRCDIR)/ShowColorMap.cc\
	$(SRCDIR)/ShowField.cc\
	$(SRCDIR)/ShowMatrix.cc\
	$(SRCDIR)/StreamLines.cc\
	$(SRCDIR)/NrrdTextureBuilder.cc\
	$(SRCDIR)/TextureBuilder.cc\
	$(SRCDIR)/VolumeSlicer.cc\
	$(SRCDIR)/VolumeVisualizer.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Dataflow/Widgets Dataflow/Ports \
	Core/Algorithms/Fields \
	Core/Datatypes Core/Containers \
	Core/Exceptions Core/Thread Core/GuiInterface Core/Geom \
	Core/Persistent Core/Geometry Core/2d Core/Util \
	Core/TkExtensions Core/Algorithms/Visualization \
	Core/GeomInterface Core/Volume

LIBS := $(TEEM_LIBRARY) $(FTGL_LIBRARY) $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

INCLUDES += $(FTGL_INCLUDE)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif
