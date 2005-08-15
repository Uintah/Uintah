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

SRCDIR   := Core/CCA/PIDL

SRCS     += \
	$(SRCDIR)/InvalidReference.cc \
	$(SRCDIR)/MalformedURL.cc \
	$(SRCDIR)/Object.cc \
	$(SRCDIR)/Object_proxy.cc \
	$(SRCDIR)/PIDL.cc \
	$(SRCDIR)/PIDLException.cc \
	$(SRCDIR)/ProxyBase.cc \
	$(SRCDIR)/Reference.cc \
	$(SRCDIR)/ServerContext.cc \
	$(SRCDIR)/URL.cc \
	$(SRCDIR)/Warehouse.cc \
	$(SRCDIR)/TypeInfo.cc \
	$(SRCDIR)/TypeInfo_internal.cc \
	$(SRCDIR)/MxNArrayRep.cc \
	$(SRCDIR)/MxNScheduler.cc \
	$(SRCDIR)/MxNScheduleEntry.cc \
	$(SRCDIR)/MxNArrSynch.cc \
	$(SRCDIR)/MxNMetaSynch.cc \
	$(SRCDIR)/HandlerStorage.cc \
	$(SRCDIR)/HandlerGateKeeper.cc \
	$(SRCDIR)/ReferenceMgr.cc  \
	$(SRCDIR)/XceptionRelay.cc




PSELIBS := Core/Exceptions Core/Thread Core/CCA/Comm/DT
LIBS := $(UUID_LIB) $(M_LIBRARY)

ifeq ($(HAVE_GLOBUS),yes)
PSELIBS += Core/globus_threads 
LIBS +=  $(GLOBUS_LIBRARY) $(GLOBUS_IO_LIBRARYK) 
endif

####################################################################
#Intra is removed before the completion of Parallel CCA Components
#ifeq ($(HAVE_MPI),yes) 
#PSELIBS += Core/CCA/Comm/Intra
#LIBS += $(MPI_LIBRARY)
#endif
####################################################################

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

