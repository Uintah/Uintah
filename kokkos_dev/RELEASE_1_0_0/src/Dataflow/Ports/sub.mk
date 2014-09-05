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

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Ports

SRCS     += $(SRCDIR)/ColorMapPort.cc             \
            $(SRCDIR)/FieldPort.cc                \
            $(SRCDIR)/FieldSetPort.cc             \
            $(SRCDIR)/GeomPort.cc                 \
            $(SRCDIR)/GeometryPort.cc             \
	    $(SRCDIR)/GLTexture3DPort.cc          \
            $(SRCDIR)/HexMeshPort.cc              \
	    $(SRCDIR)/ImagePort.cc		  \
            $(SRCDIR)/MatrixPort.cc               \
            $(SRCDIR)/PathPort.cc                 \
            $(SRCDIR)/VoidStarPort.cc             \
            $(SRCDIR)/templates.cc


PSELIBS := Dataflow/Network Dataflow/Comm Core/Containers \
	Core/Thread Core/Geom Core/Geometry Core/Exceptions \
	Core/Persistent Core/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

