#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

SRCDIR := Packages/Teem/Dataflow/GUI

SRCS := \
	$(SRCDIR)/AnalyzeNrrdReader.tcl\
	$(SRCDIR)/axis_info_sel_box.tcl\
	$(SRCDIR)/ChooseNrrd.tcl\
	$(SRCDIR)/DicomNrrdReader.tcl\
	$(SRCDIR)/FieldToNrrd.tcl\
	$(SRCDIR)/ImageImporter.tcl\
	$(SRCDIR)/ImageExporter.tcl\
	$(SRCDIR)/MRITissueClassifier.tcl\
	$(SRCDIR)/NrrdInfo.tcl\
	$(SRCDIR)/NrrdSetProperty.tcl\
	$(SRCDIR)/NrrdSetupTexture.tcl\
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
	$(SRCDIR)/TendPoint.tcl\
	$(SRCDIR)/TendSatin.tcl\
	$(SRCDIR)/Unu1op.tcl\
	$(SRCDIR)/Unu2op.tcl\
	$(SRCDIR)/Unu3op.tcl\
	$(SRCDIR)/UnuAxdelete.tcl\
	$(SRCDIR)/UnuAxinfo.tcl\
	$(SRCDIR)/UnuAxinsert.tcl\
	$(SRCDIR)/UnuAxmerge.tcl\
	$(SRCDIR)/UnuAxsplit.tcl\
	$(SRCDIR)/UnuCCadj.tcl\
	$(SRCDIR)/UnuCCfind.tcl\
	$(SRCDIR)/UnuCCmerge.tcl\
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
	$(SRCDIR)/UnuJhisto.tcl\
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
	$(SRCDIR)/GageProbe.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

TEEM_MODULES := $(TEEM_MODULES) $(TCLINDEX)

