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

SRCDIR   := Core/CCA/Comm

ifeq ($(HAVE_GLOBUS),yes)
SRCS     += \
	$(SRCDIR)/Message.cc \
	$(SRCDIR)/SocketSpChannel.cc \
	$(SRCDIR)/SocketEpChannel.cc \
	$(SRCDIR)/NexusEpChannel.cc \
	$(SRCDIR)/NexusSpChannel.cc \
	$(SRCDIR)/CommError.cc \
	$(SRCDIR)/NexusSpMessage.cc \
	$(SRCDIR)/NexusEpMessage.cc \
	$(SRCDIR)/SocketMessage.cc \
	$(SRCDIR)/SocketThread.cc \
	$(SRCDIR)/ReplyEP.cc \
	$(SRCDIR)/NexusHandlerThread.cc \
	$(SRCDIR)/CommNexus.cc
LIBS := $(GLOBUS_COMMON_LIBRARY) $(GLOBUS_LIBRARY) $(GLOBUS_IO_LIBRARY)

else

SRCS     += \
	$(SRCDIR)/Message.cc \
	$(SRCDIR)/SocketSpChannel.cc \
	$(SRCDIR)/SocketEpChannel.cc \
	$(SRCDIR)/CommError.cc \
	$(SRCDIR)/SocketMessage.cc \
	$(SRCDIR)/SocketThread.cc \
LIBS :=
endif

ifeq ($(HAVE_GLOBUS),yes)
PSELIBS := Core/Exceptions Core/Util Core/Thread Core/globus_threads
else
PSELIBS := Core/Exceptions Core/Util Core/Thread Core/CCA/Comm/DT
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(HAVE_MPI),yes)
SUBDIRS := \
        $(SRCDIR)/Intra   
endif
SUBDIRS+=$(SRCDIR)/DT
include $(SCIRUN_SCRIPTS)/recurse.mk
