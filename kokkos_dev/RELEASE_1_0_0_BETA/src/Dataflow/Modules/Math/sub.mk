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

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Math

SRCS     += \
	$(SRCDIR)/BuildTransform.cc\
	$(SRCDIR)/ErrorMetric.cc\
        $(SRCDIR)/ManipMatrix.cc\
	$(SRCDIR)/SolveMatrix.cc\
#[INSERT NEW CODE FILE HERE]

CFLAGS   := $(CFLAGS) -DMM_SRC=\"$(SRCTOP_ABS)/Dataflow/Modules/Math/Manip\" -DMM_OBJ=\"$(OBJTOP_ABS)/Dataflow/Modules/Math/Manip\"

PSELIBS := Dataflow/Network Dataflow/Ports Dataflow/XMLUtil \
	Core/Datatypes Core/Persistent \
	Core/Exceptions Core/Thread Core/Containers \
	Core/GuiInterface Core/Geometry Core/Datatypes \
	Core/Util Core/Geom Core/TkExtensions \
	Dataflow/Widgets
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm $(XML_LIBRARY) 

include $(SRCTOP)/scripts/smallso_epilogue.mk
