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

SRCDIR   := main

########################################################################
#
# scirun (aka PROGRAM_PSE)
#

SRCS    := $(SRCDIR)/main.cc

ifeq ($(LARGESOS),yes)
  PSELIBS := Dataflow Core
else
  PSELIBS := \
        Core/Basis         \
        Core/Comm          \
        Core/Containers    \
        Core/Exceptions    \
        Core/Geom          \
        Dataflow/GuiInterface  \
        Core/ICom          \
        Core/Init          \
        Core/Services      \
        Core/SystemCall    \
        Core/Thread        \
        Dataflow/TkExtensions  \
        Core/Util          \
        Core/Volume        \
        Core/XMLUtil       \
        Dataflow/Network   \
        Dataflow/TCLThread 

  ifeq ($(HAVE_PTOLEMY_PACKAGE), yes)   
        PSELIBS += Packages/Ptolemy/Core/Comm
  endif

  ifeq ($(OS_NAME),Darwin)
    PSELIBS += Core/Datatypes Core/ImportExport Core/Persistent
  endif
endif

# Though SCIRun (main.o) doesn't explicitly use libxml2, we must link
# against it (on the (perhaps only OSX 8.5) mac) so that the system
# libxml2 (reference through the Foundation classes) doesn't get called.
LIBS :=  $(XML_LIBRARY) $(XML2_LIBRARY) $(F_LIBRARY)

ifeq ($(NEED_SONAME),yes)
  LIBS := $(LIBS) $(XML_LIBRARY) $(TK_LIBRARY) $(DL_LIBRARY) $(Z_LIBRARY) $(SCISOCK_LIBRARY)
endif

PROGRAM := $(PROGRAM_PSE)

ifeq ($(OS_NAME),Darwin)
  PROGRAM_LDFLAGS := $(PROGRAM_LDFLAGS) -bind_at_load
endif

include $(SCIRUN_SCRIPTS)/program.mk


########################################################################
#
# scirunremote
#

SRCS     := $(SRCDIR)/scirunremote.cc

ifeq ($(LARGESOS),yes)
  PSELIBS := Dataflow Core
else
  PSELIBS := Dataflow/Network Core/Containers Dataflow/GuiInterface \
        Core/Thread Core/Exceptions Core/Util Dataflow/TkExtensions Core/Comm \
        Core/ICom Core/Services Core/XMLUtil Core/SystemCall Core/Init Core/Volume
  ifeq ($(OS_NAME),Darwin)
    PSELIBS += Core/Datatypes Core/ImportExport
  endif
endif

LIBS := $(LIBS) $(XML_LIBRARY)  $(DL_LIBRARY) $(Z_LIBRARY)

PROGRAM := scirunremote

ifeq ($(OS_NAME),Darwin)
  PROGRAM_LDFLAGS := $(PROGRAM_LDFLAGS) -bind_at_load
endif

include $(SCIRUN_SCRIPTS)/program.mk

########################################################################
