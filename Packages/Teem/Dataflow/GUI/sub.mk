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
	$(SRCDIR)/axis_info_sel_box.tcl\
	$(SRCDIR)/FieldToNrrd.tcl\
	$(SRCDIR)/HDF5DataReader.tcl\
	$(SRCDIR)/NrrdAxmerge.tcl\
	$(SRCDIR)/NrrdAxsplit.tcl\
	$(SRCDIR)/NrrdCmedian.tcl\
	$(SRCDIR)/NrrdConvert.tcl\
	$(SRCDIR)/NrrdCrop.tcl\
	$(SRCDIR)/NrrdDhisto.tcl\
	$(SRCDIR)/NrrdFlip.tcl\
	$(SRCDIR)/NrrdHisto.tcl\
	$(SRCDIR)/NrrdInfo.tcl\
	$(SRCDIR)/NrrdJoin.tcl\
	$(SRCDIR)/NrrdPad.tcl\
	$(SRCDIR)/NrrdPermute.tcl\
	$(SRCDIR)/NrrdProject.tcl\
	$(SRCDIR)/NrrdQuantize.tcl\
	$(SRCDIR)/NrrdReader.tcl\
	$(SRCDIR)/NrrdResample.tcl\
	$(SRCDIR)/NrrdSelectTime.tcl\
	$(SRCDIR)/NrrdSlice.tcl\
	$(SRCDIR)/NrrdWriter.tcl\
	$(SRCDIR)/TendAnvol.tcl\
	$(SRCDIR)/TendBmat.tcl\
	$(SRCDIR)/TendEpireg.tcl\
	$(SRCDIR)/TendEstim.tcl\
	$(SRCDIR)/TendEval.tcl\
	$(SRCDIR)/TendEvec.tcl\
	$(SRCDIR)/TendExpand.tcl\
	$(SRCDIR)/TendMake.tcl\
	$(SRCDIR)/TendPoint.tcl\
	$(SRCDIR)/TendSatin.tcl\
	$(SRCDIR)/TendShrink.tcl\

#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

