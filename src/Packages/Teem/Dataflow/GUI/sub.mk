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

SRCDIR := Packages/Teem/Dataflow/GUI

SRCS := \
	$(SRCDIR)/AnalyzeToNrrd.tcl\
	$(SRCDIR)/axis_info_sel_box.tcl\
	$(SRCDIR)/ChooseNrrd.tcl\
	$(SRCDIR)/DicomToNrrd.tcl\
	$(SRCDIR)/EditTupleAxis.tcl\
	$(SRCDIR)/FieldToNrrd.tcl\
	$(SRCDIR)/NrrdInfo.tcl\
	$(SRCDIR)/NrrdSetProperty.tcl\
	$(SRCDIR)/NrrdReader.tcl\
	$(SRCDIR)/NrrdSelectTime.tcl\
	$(SRCDIR)/NrrdToField.tcl\
	$(SRCDIR)/NrrdWriter.tcl\
	$(SRCDIR)/TendAnvol.tcl\
	$(SRCDIR)/TendBmat.tcl\
	$(SRCDIR)/TendEpireg.tcl\
	$(SRCDIR)/TendEstim.tcl\
	$(SRCDIR)/TendEval.tcl\
	$(SRCDIR)/TendEvec.tcl\
	$(SRCDIR)/TendExpand.tcl\
	$(SRCDIR)/TendPoint.tcl\
	$(SRCDIR)/TendSatin.tcl\
	$(SRCDIR)/TendShrink.tcl\
	$(SRCDIR)/UnuAxmerge.tcl\
	$(SRCDIR)/UnuAxsplit.tcl\
	$(SRCDIR)/UnuCmedian.tcl\
	$(SRCDIR)/UnuConvert.tcl\
	$(SRCDIR)/UnuCrop.tcl\
	$(SRCDIR)/UnuDhisto.tcl\
	$(SRCDIR)/UnuFlip.tcl\
	$(SRCDIR)/UnuHisto.tcl\
	$(SRCDIR)/UnuJoin.tcl\
	$(SRCDIR)/UnuPad.tcl\
	$(SRCDIR)/UnuPermute.tcl\
	$(SRCDIR)/UnuProject.tcl\
	$(SRCDIR)/UnuQuantize.tcl\
	$(SRCDIR)/UnuResample.tcl\
	$(SRCDIR)/UnuSlice.tcl\

#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

TEEM_MODULES := $(TEEM_MODULES) $(TCLINDEX)

