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
# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Packages/CardioWave/Dataflow/GUI

SRCS := \
	$(SRCDIR)/CreateParametersBundle.tcl \
	$(SRCDIR)/CreateSimpleMesh.tcl \
	$(SRCDIR)/HexIntMask.tcl \
	$(SRCDIR)/ReclassifyInteriorTets.tcl \
	$(SRCDIR)/RemoveInteriorTets.tcl \
	$(SRCDIR)/SetupFVMatrix.tcl \
	$(SRCDIR)/SetupFVM2.tcl \
	$(SRCDIR)/TimeDataReader.tcl\
	$(SRCDIR)/ReduceBandWidth.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

