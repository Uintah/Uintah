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

RCDIR   := main
SRCS    := $(SRCDIR)/main.cc


ifeq ($(LARGESOS),yes)
  PSELIBS := Dataflow Core
else
  PSELIBS := Dataflow/Network Core/Containers Dataflow/TCLThread Core/GuiInterface \
        Core/Thread Core/Exceptions Core/Util Core/TkExtensions Core/Comm \
        Core/ICom Core/Services Core/XMLUtil Core/SystemCall Core/Geom Core/Init
  ifeq ($(HAVE_PTOLEMY), yes)   
        PSELIBS += Packages/Ptolemy/Core/Comm
  endif
  ifeq ($(OS_NAME),Darwin)
    PSELIBS += Core/Datatypes Core/ImportExport Core/Persistent
  endif
endif

LIBS :=  $(XML_LIBRARY)
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
# SCIRun2 Stuff:
#

ifeq ($(BUILD_SCIRUN2),yes)

  ########################################################################
  #
  # sr
  #
  SRCS      := $(SRCDIR)/newmain.cc

  ifeq ($(LARGESOS),yes)
    PSELIBS := Core/CCA
  else
    PSELIBS := Core/Exceptions Core/CCA/Comm    \
        Core/CCA/PIDL Core/CCA/spec Core/Util \
        SCIRun Core/CCA/SSIDL Core/Thread
    ifeq ($(HAVE_GLOBUS),yes)   
      PSELIBS += Core/globus_threads
    endif
  endif

  LIBS := $(GLOBUS_LIBRARY)

  ifeq ($(HAVE_MPI),yes)
    LIBS += $(MPI_LIBRARY) 
  endif

  PROGRAM := sr

  include $(SCIRUN_SCRIPTS)/program.mk

  ########################################################################
  #
  # ploader
  #

  #build the SCIRun CCA Component Loader here
  ifeq ($(LARGESOS),yes)
    PSELIBS := Core/CCA/Component
  else

    ifeq ($(HAVE_GLOBUS),yes)
      PSELIBS := Core/Exceptions Core/CCA/Comm Core/CCA/Comm/DT \
        Core/CCA/PIDL Core/globus_threads Core/CCA/spec \
        SCIRun Core/CCA/SSIDL Core/Thread 
    else
      PSELIBS := Core/Exceptions Core/CCA/Comm\
        Core/CCA/PIDL Core/CCA/spec \
        SCIRun Core/CCA/SSIDL Core/Thread 
    endif
  endif

  ifeq ($(HAVE_MPI),yes)
    LIBS := $(MPI_LIBRARY) 
  endif

  PROGRAM := ploader
  SRCS      := $(SRCDIR)/ploader.cc
  include $(SCIRUN_SCRIPTS)/program.mk

endif # Build SCIRun2

########################################################################
#
# scirunremote
#

SRCS     := $(SRCDIR)/scirunremote.cc

ifeq ($(LARGESOS),yes)
  PSELIBS := Dataflow Core
else
  PSELIBS := Dataflow/Network Core/Containers Core/GuiInterface \
        Core/Thread Core/Exceptions Core/Util Core/TkExtensions Core/Comm \
        Core/ICom Core/Services Core/XMLUtil Core/SystemCall Core/Init
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

