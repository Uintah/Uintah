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

SRCDIR   := Dataflow/Modules/Render

SRCS     += \
	$(SRCDIR)/Ball.cc\
	$(SRCDIR)/BallAux.cc\
	$(SRCDIR)/BallMath.cc\
	$(SRCDIR)/Camera.cc\
	$(SRCDIR)/EditPath.cc\
	$(SRCDIR)/OpenGL.cc\
	$(SRCDIR)/SynchronizeGeometry.cc\
	$(SRCDIR)/ViewWindow.cc\
	$(SRCDIR)/ViewSlices.cc\
	$(SRCDIR)/Painter.cc\
	$(SRCDIR)/PainterTools.cc\
	$(SRCDIR)/PainterBrushTool.cc\
	$(SRCDIR)/Viewer.cc\
	$(SRCDIR)/glMath.cc\
#[INSERT NEW CODE FILE HERE]


PSELIBS := \
	Core/Algorithms/Fields \
	Core/Basis         \
	Core/Bundle        \
	Core/Datatypes     \
	Core/Exceptions    \
	Core/Geometry      \
	Core/Geom          \
	Core/GeomInterface \
	Core/Containers    \
	Core/GuiInterface  \
	Core/Persistent    \
	Core/Thread        \
	Core/TkExtensions  \
	Core/Util          \
	Core/Volume        \
	Dataflow/Comm      \
	Dataflow/Network   \
	Dataflow/Widgets 

INCLUDES += $(MPEG_INCLUDE) $(MAGICK_INCLUDE) $(TEEM_INCLUDE)

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(MPEG_LIBRARY) $(MAGICK_LIBRARY) $(M_LIBRARY) $(TEEM_LIBRARY)

# CollabVis code begin
ifeq ($(HAVE_COLLAB_VIS),yes)
  SRCS += $(SRCTOP)/Packages/CollabVis/Standalone/Server/ViewServer.cc

  # SUBDIRS += $(SRCDIR)/SV_Server

  INCLUDES += -I$(SRCTOP)/Packages/CollabVis/Standalone/Server

  REMOTELIBS := -L$(SRCTOP)/Packages/CollabVis/Standalone/Server/lib \
	-lXML \
	-lCompression\
	-lExceptions \
	-lLogging \
	-lMessage \
	-lNetwork \
	-lProperties \
	-lRendering \
	-lThread 

  LIBS += $(XML_LIBRARY) $(REMOTELIBS) -lm
endif
# CollabVis code end

ifeq ($(NEED_OSX_SYSTEMSTUBS),yes)
  LIBS += -lSystemStubs
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif
