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


# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/Builder

SRCS     += \
	$(SRCDIR)/Builder.cc $(SRCDIR)/BuilderWindow.cc \
	$(SRCDIR)/QtUtils.cc $(SRCDIR)/NetworkCanvasView.cc \
	$(SRCDIR)/Module.cc $(SRCDIR)/PortIcon.cc \
	$(SRCDIR)/ClusterDialog.cc $(SRCDIR)/PathDialog.cc \
	$(SRCDIR)/Connection.cc $(SRCDIR)/BridgeConnection.cc\
	$(SRCDIR)/moc_BuilderWindow.cc $(SRCDIR)/moc_NetworkCanvasView.cc \
	$(SRCDIR)/moc_Module.cc $(SRCDIR)/moc_ClusterDialog.cc \
	$(SRCDIR)/moc_PathDialog.cc

PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/CCA/Comm Core/CCA/spec \
			Core/Thread Core/Containers Core/Exceptions Core/Util

LIBS := $(QT_LIBRARY) $(LIBS)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

PROGRAM := builder_main
SRCS := $(SRCDIR)/builder_main.cc
PSELIBS := CCA/Components/Builder Core/CCA/SSIDL \
	Core/CCA/PIDL Core/Exceptions Core/CCA/spec SCIRun

LIBS := $(QT_LIBRARY) $(LIBS)

include $(SCIRUN_SCRIPTS)/program.mk

$(SRCDIR)/Builder.o: Core/CCA/spec/sci_sidl.h
$(SRCDIR)/BuilderWindow.o: Core/CCA/spec/sci_sidl.h
$(SRCDIR)/moc_BuilderWindow.o: Core/CCA/spec/sci_sidl.h
$(SRCDIR)/builder_main.o: Core/CCA/spec/sci_sidl.h
