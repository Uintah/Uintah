#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/CCA/Component/PIDL

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
	$(SRCDIR)/HandlerStorage.cc \
	$(SRCDIR)/HandlerGateKeeper.cc \
	$(SRCDIR)/ReferenceMgr.cc




ifeq ($(HAVE_GLOBUS),yes)
PSELIBS := Core/Exceptions Core/Thread Core/globus_threads Core/CCA/Component/Comm/Intra
LIBS := $(GLOBUS_LIBRARY) $(GLOBUS_IO_LIBRARYK) $(MPI_LIBRARY) $(UUID_LIB) $(M_LIBRARY)
else
PSELIBS := Core/Exceptions Core/Thread Core/CCA/Component/Comm/Intra
LIBS := $(MPI_LIBRARY) $(UUID_LIB) $(M_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

