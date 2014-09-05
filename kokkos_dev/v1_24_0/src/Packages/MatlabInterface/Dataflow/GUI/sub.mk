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

SRCDIR := Packages/MatlabInterface/Dataflow/GUI

SRCS := \
	$(SRCDIR)/Matlab.tcl\
	$(SRCDIR)/MatlabBundle.tcl\
	$(SRCDIR)/MatlabDataReader.tcl\
	$(SRCDIR)/MatlabDataWriter.tcl\
	$(SRCDIR)/MatlabFieldsReader.tcl\
	$(SRCDIR)/MatlabFieldsWriter.tcl\
	$(SRCDIR)/MatlabMatricesReader.tcl\
	$(SRCDIR)/MatlabNrrdsReader.tcl\
	$(SRCDIR)/MatlabNrrdsWriter.tcl\
	$(SRCDIR)/MatlabMatricesWriter.tcl\
	$(SRCDIR)/MatlabBundlesWriter.tcl\
	$(SRCDIR)/MatlabBundlesReader.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

MATLABIO_MODULES := $(MATLABIO_MODULES) $(TCLINDEX)

