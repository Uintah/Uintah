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

SRCDIR   := SCIRun

SRCS     += \
            $(SRCDIR)/SCIRunFramework.cc \
            $(SRCDIR)/ComponentDescription.cc \
            $(SRCDIR)/ComponentInstance.cc \
            $(SRCDIR)/ComponentModel.cc \
            $(SRCDIR)/PortDescription.cc \
            $(SRCDIR)/SCIRunErrorHandler.cc \
            $(SRCDIR)/PortInstance.cc \
            $(SRCDIR)/PortInstanceIterator.cc\
            $(SRCDIR)/CCACommunicator.cc \
            $(SRCDIR)/resourceReference.cc \
            $(SRCDIR)/TypeMap.cc \
            $(SRCDIR)/SCIRunLoader.cc


SUBDIRS := $(SRCDIR)/CCA $(SRCDIR)/Dataflow $(SRCDIR)/Internal \
           $(SRCDIR)/Corba $(SRCDIR)/Tao

ifeq ($(HAVE_VTK),yes)
 SUBDIRS += $(SRCDIR)/Vtk
endif

ifeq ($(HAVE_BABEL),yes)
 ifeq ($(HAVE_RUBY),yes)
  SUBDIRS += $(SRCDIR)/Bridge
 endif
 SUBDIRS += $(SRCDIR)/Babel
endif

include $(SCIRUN_SCRIPTS)/recurse.mk
ifeq ($(HAVE_GLOBUS),yes)
 PSELIBS := Core/OS Core/Containers Core/Util Dataflow/XMLUtil \
            Dataflow/Network Core/GuiInterface Core/CCA/spec \
            Core/CCA/PIDL Core/CCA/SSIDL \
            Core/Exceptions Core/TkExtensions Dataflow/TCLThread Core/Init Core/Thread \
            Core/globus_threads Core/CCA/Comm
else
 PSELIBS := Core/OS Core/Containers Core/Util Dataflow/XMLUtil \
            Dataflow/Network Core/GuiInterface Core/CCA/spec \
            Core/CCA/PIDL Core/CCA/SSIDL \
            Core/Exceptions Dataflow/Thread \
            Core/TkExtensions Core/TCLThread Core/Init Core/CCA/Comm
endif

LIBS := $(XML_LIBRARY)

ifeq ($(HAVE_RUBY),yes)
 PSELIBS := $(PSELIBS) Core/CCA/tools/strauss
 INCLUDES := $(RUBY_INCLUDE) $(INCLUDES) 
 LIBS := $(RUBY_LIBRARY) $(LIBS)
endif

ifeq ($(HAVE_MPI),yes)
 LIBS := $(LIBS) $(MPI_LIBRARY)
endif

LIBS := $(LIBS) -lcrypt -ldl

ifeq ($(HAVE_BABEL),yes)
  LIBS := $(LIBS) $(SIDL_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
