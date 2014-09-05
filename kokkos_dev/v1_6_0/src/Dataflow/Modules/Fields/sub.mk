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
	$(SRCDIR)/AttractNormals.cc\
	$(SRCDIR)/BuildInterpolant.cc\
	$(SRCDIR)/CastMLVtoHV.cc\
	$(SRCDIR)/CastTVtoMLV.cc\
	$(SRCDIR)/Centroids.cc\
	$(SRCDIR)/ChangeFieldDataAt.cc\
	$(SRCDIR)/ChangeFieldDataType.cc\
	$(SRCDIR)/ChangeFieldBounds.cc\
	$(SRCDIR)/ClipField.cc\
	$(SRCDIR)/ClipLattice.cc\
	$(SRCDIR)/ConvertTet.cc\
	$(SRCDIR)/Coregister.cc\
	$(SRCDIR)/DirectInterpolate.cc\
	$(SRCDIR)/EditField.cc\
	$(SRCDIR)/FieldBoundary.cc\
	$(SRCDIR)/FieldInfo.cc\
	$(SRCDIR)/FieldMeasures.cc\
	$(SRCDIR)/GatherPoints.cc\
	$(SRCDIR)/Gradient.cc\
	$(SRCDIR)/ManageFieldData.cc\
	$(SRCDIR)/Probe.cc\
	$(SRCDIR)/SampleField.cc\
	$(SRCDIR)/SampleLattice.cc\
	$(SRCDIR)/SamplePlane.cc\
	$(SRCDIR)/ScalarFieldStats.cc\
	$(SRCDIR)/ScaleFieldData.cc\
	$(SRCDIR)/SelectElements.cc\
	$(SRCDIR)/SelectField.cc\
	$(SRCDIR)/SetProperty.cc\
	$(SRCDIR)/TetVolFieldCellToNode.cc\
	$(SRCDIR)/TetVol2QuadraticTetVol.cc\
	$(SRCDIR)/TransformField.cc\
	$(SRCDIR)/TransformScalarData.cc\
	$(SRCDIR)/Unstructure.cc\
	$(SRCDIR)/VectorMagnitude.cc\
#[INSERT NEW CODE FILE HERE]
#	$(SRCDIR)/CastField.cc\
#	$(SRCDIR)/ChangeCellType.cc\
#	$(SRCDIR)/ManipFields.cc\


PSELIBS := Dataflow/Network Dataflow/Ports Dataflow/XMLUtil Dataflow/Widgets \
	Core/Datatypes Core/Persistent Core/Exceptions \
	Core/Thread Core/Containers Core/GuiInterface Core/Geom \
	Core/Datatypes Core/Geometry Core/TkExtensions \
	Core/Math Core/Util Core/Algorithms/Geometry Core/GeomInterface
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(FLEX_LIBS) -lm $(XML_LIBRARY) $(THREAD_LIBS)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

