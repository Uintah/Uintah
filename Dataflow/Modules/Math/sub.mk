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

SRCDIR   := Dataflow/Modules/Math

SRCS     += \
	$(SRCDIR)/AppendMatrix.cc\
	$(SRCDIR)/BuildNoise.cc\
	$(SRCDIR)/BuildTransform.cc\
	$(SRCDIR)/CastMatrix.cc\
	$(SRCDIR)/ErrorMetric.cc\
	$(SRCDIR)/FieldToCanonicalTransform.cc\
	$(SRCDIR)/LinAlgBinary.cc\
	$(SRCDIR)/LinAlgUnary.cc\
	$(SRCDIR)/LinearAlgebra.cc\
        $(SRCDIR)/MatrixSelectVector.cc\
        $(SRCDIR)/MinNormLeastSq.cc\
	$(SRCDIR)/SolveMatrix.cc\
	$(SRCDIR)/Submatrix.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Dataflow/Ports Dataflow/XMLUtil \
	Core/Datatypes Core/Persistent Core/Math \
	Core/Exceptions Core/Thread Core/Containers \
	Core/GuiInterface Core/Geometry Core/Datatypes \
	Core/Util Core/Geom Core/TkExtensions Core/GeomInterface \
	Dataflow/Widgets
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(XML_LIBRARY) $(PETSC_UNI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
