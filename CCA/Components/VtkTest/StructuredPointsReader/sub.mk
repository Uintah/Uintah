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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/VtkTest/StructuredPointsReader

SRCS     += $(SRCDIR)/StructuredPointsReader.cc
PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/CCA/Comm\
	Core/CCA/spec Core/Thread Core/Containers Core/Exceptions
QT_LIBDIR := /home/sparker/SCIRun/SCIRun_Thirdparty_32_linux/lib
LIBS := $(QT_LIBRARY)

Vtk_LIBRARY:= -L/home/kzhang/download/VTK/bin \
	 -lvtkFiltering -lvtkGraphics -lvtkImaging -lvtkCommon -lvtkRendering -lvtkIO

LIBS += $(Vtk_LIBRARY)

INCLUDES+=\
	 -I/home/kzhang/download/VTK \
	 -I/home/kzhang/download/VTK/Imaging\
	 -I/home/kzhang/download/VTK/Graphics \
	 -I/home/kzhang/download/VTK/Rendering \
	 -I/home/kzhang/download/VTK/Filtering \
	 -I/home/kzhang/download/VTK/Common \
	 -I/home/kzhang/download/VTK/IO 


include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk




