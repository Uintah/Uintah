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

SRCDIR   := Dataflow/Modules/DataIO

SRCS     += \
	$(SRCDIR)/ColorMapReader.cc\
	$(SRCDIR)/ColorMapWriter.cc\
	$(SRCDIR)/FieldReader.cc\
	$(SRCDIR)/FieldWriter.cc\
	$(SRCDIR)/MatrixReader.cc\
	$(SRCDIR)/MatrixWriter.cc\
	$(SRCDIR)/PathReader.cc\
	$(SRCDIR)/PathWriter.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Dataflow/Ports Core/Datatypes \
	Core/Persistent Core/Exceptions Core/Thread Core/Containers \
	Core/GuiInterface Core/Geometry Core/Datatypes \
	Core/Util Core/Geom Core/TkExtensions Core/GeomInterface \
	Dataflow/Widgets
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif
