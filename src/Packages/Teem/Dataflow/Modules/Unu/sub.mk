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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Teem/Dataflow/Modules/Unu


SRCS     += \
	$(SRCDIR)/Unu1op.cc\
	$(SRCDIR)/Unu2op.cc\
	$(SRCDIR)/Unu3op.cc\
	$(SRCDIR)/UnuAxdelete.cc\
	$(SRCDIR)/UnuAxinfo.cc\
	$(SRCDIR)/UnuAxinsert.cc\
	$(SRCDIR)/UnuAxmerge.cc\
	$(SRCDIR)/UnuAxsplit.cc\
	$(SRCDIR)/UnuCCadj.cc\
	$(SRCDIR)/UnuCCfind.cc\
	$(SRCDIR)/UnuCCmerge.cc\
	$(SRCDIR)/UnuCCsettle.cc\
	$(SRCDIR)/UnuCmedian.cc\
	$(SRCDIR)/UnuConvert.cc\
	$(SRCDIR)/UnuCrop.cc\
	$(SRCDIR)/UnuDhisto.cc\
	$(SRCDIR)/UnuFlip.cc\
	$(SRCDIR)/UnuGamma.cc\
	$(SRCDIR)/UnuHeq.cc\
	$(SRCDIR)/UnuHistax.cc\
	$(SRCDIR)/UnuHisto.cc\
	$(SRCDIR)/UnuImap.cc\
	$(SRCDIR)/UnuInset.cc\
	$(SRCDIR)/UnuJhisto.cc\
	$(SRCDIR)/UnuJoin.cc\
	$(SRCDIR)/UnuLut.cc\
	$(SRCDIR)/UnuMake.cc\
	$(SRCDIR)/UnuMinmax.cc\
	$(SRCDIR)/UnuPad.cc\
	$(SRCDIR)/UnuPermute.cc\
	$(SRCDIR)/UnuProject.cc\
	$(SRCDIR)/UnuQuantize.cc\
	$(SRCDIR)/UnuResample.cc\
	$(SRCDIR)/UnuReshape.cc\
	$(SRCDIR)/UnuRmap.cc\
	$(SRCDIR)/UnuSave.cc\
	$(SRCDIR)/UnuSlice.cc\
	$(SRCDIR)/UnuSplice.cc\
	$(SRCDIR)/UnuShuffle.cc\
	$(SRCDIR)/UnuSwap.cc\
	$(SRCDIR)/UnuUnquantize.cc\
#[INSERT NEW CODE FILE HERE]

# the following are stubs that need implementa


PSELIBS := Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/GeomInterface Core/Datatypes Core/Geometry \
        Core/TkExtensions Dataflow/Network Dataflow/Ports

LIBS := $(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
TEEM_MODULES := $(TEEM_MODULES) $(LIBNAME)
endif
