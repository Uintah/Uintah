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

SRCDIR   := Dataflow/Modules/Fields

SRCS     += \
	$(SRCDIR)/ApplyInterpolant.cc\
	$(SRCDIR)/BuildInterpolant.cc\
	$(SRCDIR)/CastMLVtoHV.cc\
	$(SRCDIR)/CastTVtoMLV.cc\
	$(SRCDIR)/ClippingPlane.cc\
	$(SRCDIR)/DirectInterpolate.cc\
	$(SRCDIR)/DirectInterpolateAlgo.cc\
	$(SRCDIR)/EditField.cc\
	$(SRCDIR)/FieldBoundary.cc\
	$(SRCDIR)/Gradient.cc\
	$(SRCDIR)/ManageFieldData.cc\
	$(SRCDIR)/ManageFieldSet.cc\
	$(SRCDIR)/ScaleFieldData.cc\
	$(SRCDIR)/SeedField.cc\
	$(SRCDIR)/SelectField.cc\
	$(SRCDIR)/TransformField.cc\
	$(SRCDIR)/TetVolCellToNode.cc\
#[INSERT NEW CODE FILE HERE]
#	$(SRCDIR)/CastField.cc\
#	$(SRCDIR)/ChangeCellType.cc\
#	$(SRCDIR)/ManipFields.cc\


PSELIBS := Dataflow/Network Dataflow/Ports Dataflow/XMLUtil Dataflow/Widgets \
	Core/Datatypes Core/Disclosure Core/Persistent Core/Exceptions \
	Core/Thread Core/Containers Core/GuiInterface Core/Geom \
	Core/Datatypes Core/Geometry Core/TkExtensions \
	Core/Math Core/Util
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(FLEX_LIBS) -lm $(XML_LIBRARY) $(THREAD_LIBS)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

