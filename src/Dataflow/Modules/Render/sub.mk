#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
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
	$(SRCDIR)/BaWGL.cc\
	$(SRCDIR)/Ball.cc\
	$(SRCDIR)/BallAux.cc\
	$(SRCDIR)/BallMath.cc\
	$(SRCDIR)/EditPath.cc\
	$(SRCDIR)/OpenGL.cc\
	$(SRCDIR)/Parser.cc\
	$(SRCDIR)/PBuffer.cc \
	$(SRCDIR)/SCIBaWGL.cc\
	$(SRCDIR)/SharedMemory.cc\
	$(SRCDIR)/SynchronizeGeometry.cc\
	$(SRCDIR)/Tex.cc\
	$(SRCDIR)/ViewGeom.cc\
	$(SRCDIR)/ViewWindow.cc\
	$(SRCDIR)/Viewer.cc\
	$(SRCDIR)/glMath.cc\
#[INSERT NEW CODE FILE HERE]


PSELIBS := Dataflow/Widgets Dataflow/Network Dataflow/Ports Core/Datatypes \
	Dataflow/Comm Core/Persistent Core/Exceptions Core/Geometry \
	Core/Geom Core/Thread Core/Containers \
	Core/GuiInterface Core/TkExtensions Core/Util \
	Core/TkExtensions Core/Datatypes Core/GeomInterface

INCLUDES += $(MPEG_INCLUDE) $(MAGICK_INCLUDE)

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(MPEG_LIBRARY) $(MAGICK_LIBRARY) $(M_LIBRARY)

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


include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif
