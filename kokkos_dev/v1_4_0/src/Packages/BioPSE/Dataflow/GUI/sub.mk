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
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

SRCDIR := Packages/BioPSE/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/ApplyFEMCurrentSource.tcl\
	$(SRCDIR)/ApplyFEMVoltageSource.tcl\
	$(SRCDIR)/BuildFEMatrixQuadratic.tcl\
	$(SRCDIR)/ConductivitySearch.tcl\
	$(SRCDIR)/ConfigureElectrode.tcl\
	$(SRCDIR)/DipoleSearch.tcl\
	$(SRCDIR)/InsertVoltageSource.tcl\
	$(SRCDIR)/ModifyConductivities.tcl\
	$(SRCDIR)/SetupFEMatrix.tcl\
	$(SRCDIR)/ShowDipoles.tcl\
	$(SRCDIR)/ShowLeads.tcl\
	$(SRCDIR)/Tikhonov.tcl\
	$(SRCDIR)/RawToDenseMatrix.tcl\
#[INSERT NEW TCL FILE HERE]

	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/BioPSE/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


