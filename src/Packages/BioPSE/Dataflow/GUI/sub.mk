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

SRCS := \
	$(SRCDIR)/AnisoSphereModel.tcl\
	$(SRCDIR)/ApplyFEMCurrentSource.tcl\
	$(SRCDIR)/ApplyFEMVoltageSource.tcl\
	$(SRCDIR)/BuildFEMatrixQuadratic.tcl\
	$(SRCDIR)/BuildMisfitField.tcl\
	$(SRCDIR)/ConductivitySearch.tcl\
	$(SRCDIR)/ConfigureElectrode.tcl\
	$(SRCDIR)/ConfigureWireElectrode.tcl\
	$(SRCDIR)/DipoleInAnisoSpheres.tcl\
	$(SRCDIR)/DipoleSearch.tcl\
	$(SRCDIR)/ExtractSingleSurface.tcl\
	$(SRCDIR)/InsertVoltageSource.tcl\
	$(SRCDIR)/IntegrateCurrent.tcl\
	$(SRCDIR)/ModifyConductivities.tcl\
	$(SRCDIR)/RawToDenseMatrix.tcl\
	$(SRCDIR)/SegFieldOps.tcl \
	$(SRCDIR)/SegFieldToLatVol.tcl \
	$(SRCDIR)/SepSurfToQuadSurf.tcl \
	$(SRCDIR)/SetupFEMatrix.tcl\
	$(SRCDIR)/ShowDipoles.tcl\
	$(SRCDIR)/ShowLeads.tcl\
	$(SRCDIR)/TSVD.tcl\
	$(SRCDIR)/Tikhonov.tcl\
	$(SRCDIR)/TikhonovSVD.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk




