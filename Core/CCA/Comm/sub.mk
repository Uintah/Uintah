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
	$(SRCDIR)/PRMI.cc
LIBS :=
endif

ifeq ($(HAVE_GLOBUS),yes)
PSELIBS := Core/Exceptions Core/Util Core/Thread Core/globus_threads
else
PSELIBS := Core/Exceptions Core/Util Core/Thread Core/CCA/Comm/DT
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(HAVE_MPI),yes)
SUBDIRS := $(SRCDIR)/DT
endif

include $(SCIRUN_SCRIPTS)/recurse.mk
