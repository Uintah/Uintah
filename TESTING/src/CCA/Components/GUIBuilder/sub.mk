#
#  For more information, please see: http://software.sci.utah.edu
#
#  The MIT License
#
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
#
#  
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

SRCDIR  := CCA/Components/GUIBuilder

SRCS    += \
           $(SRCDIR)/wxSCIRunApp.cc \
           $(SRCDIR)/GUIBuilder.cc \
           $(SRCDIR)/BuilderWindow.cc \
           $(SRCDIR)/NetworkCanvas.cc \
           $(SRCDIR)/MiniCanvas.cc \
           $(SRCDIR)/Connection.cc \
           $(SRCDIR)/BridgeConnection.cc \
           $(SRCDIR)/ComponentIcon.cc \
           $(SRCDIR)/PortIcon.cc \
           $(SRCDIR)/ComponentWizardDialog.cc \
           $(SRCDIR)/XMLPathDialog.cc \
           $(SRCDIR)/FrameworkProxyDialog.cc \
           $(SRCDIR)/CodePreviewDialog.cc \
           $(SRCDIR)/ComponentWizardHelper.cc 

PSELIBS := \
           SCIRun Core/CCA/spec Core/CCA/PIDL \
           Core/OS Core/Thread Core/Exceptions

LIBS := $(WX_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
