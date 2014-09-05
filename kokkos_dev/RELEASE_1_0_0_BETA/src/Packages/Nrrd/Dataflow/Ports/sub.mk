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

SRCDIR   := Packages/Nrrd/Dataflow/Ports

SRCS     += $(SRCDIR)/NrrdPort.cc  \
#[INSERT NEW CODE FILE HERE]



PSELIBS := Dataflow/Network Dataflow/Ports Core/Containers \
	Core/Thread Core/Geom Core/Geometry Core/Exceptions \
	Core/Persistent Core/Datatypes Core/Util \
	Packages/Nrrd/Core/Datatypes

LIBS := $(NRRD_LIBRARY) -lnrrd -lbiff -lair

include $(SRCTOP)/scripts/smallso_epilogue.mk

