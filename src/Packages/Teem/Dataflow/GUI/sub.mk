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
	$(SRCDIR)/FieldToNrrd.tcl\
	$(SRCDIR)/MRITissueClassifier.tcl\
	$(SRCDIR)/NrrdInfo.tcl\
	$(SRCDIR)/NrrdSetProperty.tcl\
	$(SRCDIR)/NrrdReader.tcl\
	$(SRCDIR)/NrrdSelectTime.tcl\
	$(SRCDIR)/NrrdToField.tcl\
	$(SRCDIR)/NrrdWriter.tcl\
	$(SRCDIR)/TendAnscale.tcl\
	$(SRCDIR)/TendAnvol.tcl\
	$(SRCDIR)/TendBmat.tcl\
	$(SRCDIR)/TendEpireg.tcl\
	$(SRCDIR)/TendEstim.tcl\
	$(SRCDIR)/TendEval.tcl\
	$(SRCDIR)/TendEvalClamp.tcl\
	$(SRCDIR)/TendEvalPow.tcl\
	$(SRCDIR)/TendEvec.tcl\
	$(SRCDIR)/TendEvecRGB.tcl\
	$(SRCDIR)/TendExpand.tcl\
	$(SRCDIR)/TendFiber.tcl\
	$(SRCDIR)/TendNorm.tcl\
	$(SRCDIR)/TendSatin.tcl\
	$(SRCDIR)/Unu1op.tcl\
	$(SRCDIR)/Unu2op.tcl\
	$(SRCDIR)/UnuAxdelete.tcl\
	$(SRCDIR)/UnuAxinfo.tcl\
	$(SRCDIR)/UnuAxinsert.tcl\
	$(SRCDIR)/UnuAxmerge.tcl\
	$(SRCDIR)/UnuAxsplit.tcl\
	$(SRCDIR)/UnuCCadj.tcl\
	$(SRCDIR)/UnuCCfind.tcl\
	$(SRCDIR)/UnuCmedian.tcl\
	$(SRCDIR)/UnuConvert.tcl\
	$(SRCDIR)/UnuCrop.tcl\
	$(SRCDIR)/UnuDhisto.tcl\
	$(SRCDIR)/UnuFlip.tcl\
	$(SRCDIR)/UnuGamma.tcl\
	$(SRCDIR)/UnuHeq.tcl\
	$(SRCDIR)/UnuHistax.tcl\
	$(SRCDIR)/UnuHisto.tcl\
	$(SRCDIR)/UnuImap.tcl\
	$(SRCDIR)/UnuInset.tcl\
	$(SRCDIR)/UnuJoin.tcl\
	$(SRCDIR)/UnuLut.tcl\
	$(SRCDIR)/UnuMake.tcl\
	$(SRCDIR)/UnuMinmax.tcl\
	$(SRCDIR)/UnuPad.tcl\
	$(SRCDIR)/UnuPermute.tcl\
	$(SRCDIR)/UnuProject.tcl\
	$(SRCDIR)/UnuQuantize.tcl\
	$(SRCDIR)/UnuResample.tcl\
	$(SRCDIR)/UnuReshape.tcl\
	$(SRCDIR)/UnuRmap.tcl\
	$(SRCDIR)/UnuSave.tcl\
	$(SRCDIR)/UnuSlice.tcl\
	$(SRCDIR)/UnuSplice.tcl\
	$(SRCDIR)/UnuShuffle.tcl\
	$(SRCDIR)/UnuSwap.tcl\
	$(SRCDIR)/NrrdToMatrix.tcl\
	$(SRCDIR)/UnuUnquantize.tcl\
	$(SRCDIR)/TendSim.tcl\
	$(SRCDIR)/TendSten.tcl\
	$(SRCDIR)/TendAnplot.tcl\
	$(SRCDIR)/TendAnhist.tcl\
	$(SRCDIR)/TendSlice.tcl\
	$(SRCDIR)/TendEvalAdd.tcl\
	$(SRCDIR)/TendEvq.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

TEEM_MODULES := $(TEEM_MODULES) $(TCLINDEX)

