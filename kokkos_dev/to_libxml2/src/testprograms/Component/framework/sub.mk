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

#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := testprograms/Component/framework

ifeq ($(LARGESOS),yes)
 PSELIBS := Core
else
 PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/Thread \
            Core/Exceptions Core/CCA/Comm Core/Containers
endif

ifeq ($(HAVE_GLOBUS),yes)
 PSELIBS += Core/globus_threads 
 LIBS := $(GLOBUS_LIBRARY)
else
 LIBS :=
endif

ifeq ($(HAVE_MPI),yes)
 LIBS += $(MPI_LIBRARY) 
endif

PROGRAM := $(SRCDIR)/main

SRCS    := \
	$(SRCDIR)/cca_sidl.cc \
	$(SRCDIR)/main.cc \
	$(SRCDIR)/cca.cc \
	$(SRCDIR)/Registry.cc \
	$(SRCDIR)/ComponentIdImpl.cc \
	$(SRCDIR)/ComponentImpl.cc \
	$(SRCDIR)/PortInfoImpl.cc \
	$(SRCDIR)/PortImpl.cc \
	$(SRCDIR)/FrameworkImpl.cc \
	$(SRCDIR)/SciServicesImpl.cc \
	$(SRCDIR)/BuilderServicesImpl.cc \
	$(SRCDIR)/RegistryServicesImpl.cc \
	$(SRCDIR)/LocalFramework.cc \
	$(SRCDIR)/TestPortImpl.cc 

GENHDRS := $(SRCDIR)/cca_sidl.h

# Must be after the SRCS so that the REI files can be added to them.
SUBDIRS := $(SRCDIR)/Builders $(SRCDIR)/REI $(SRCDIR)/TestComponents
include $(SCIRUN_SCRIPTS)/recurse.mk

include $(SRCTOP)/scripts/program.mk

