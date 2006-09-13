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

SRCDIR := Framework/StandAlone

########################################################################
#
# SCIRun2 Stuff:
#


########################################################################
#
# sr
#

ifeq ($(LARGESOS),yes)
  PSELIBS := Core/CCA
else
  PSELIBS := \
             Framework Core/CCA/PIDL Core/CCA/spec Core/CCA/SSIDL \
             Core/Containers Core/Exceptions Core/Thread Core/Util
  ifeq ($(HAVE_GLOBUS),yes)
	  PSELIBS += Core/globus_threads
  endif
endif

LIBS := $(GLOBUS_LIBRARY)

ifeq ($(HAVE_MPI),yes)
  LIBS += $(MPI_LIBRARY)
endif

ifeq ($(HAVE_WX),yes)
  LIBS += $(WX_LIBRARY)
endif

PROGRAM := sr
SRCS := $(SRCDIR)/main.cc
include $(SCIRUN_SCRIPTS)/program.mk

########################################################################
#
# ploader
#

# build the SCIRun2 CCA Component Loader here
ifeq ($(LARGESOS),yes)
  PSELIBS := Core/CCA/Component
else
  PSELIBS := \
             Framework Core/CCA/PIDL Core/CCA/DT Core/CCA/spec Core/CCA/SSIDL \
             Core/Exceptions Core/Thread
  ifeq ($(HAVE_GLOBUS),yes)
	PSELIBS += Core/globus_threads
  endif
endif

ifeq ($(HAVE_MPI),yes)
  LIBS := $(MPI_LIBRARY)
endif

PROGRAM := ploader
SRCS := $(SRCDIR)/ploader.cc
include $(SCIRUN_SCRIPTS)/program.mk

########################################################################
