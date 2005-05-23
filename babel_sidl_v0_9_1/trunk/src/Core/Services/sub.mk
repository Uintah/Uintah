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

#   File   : sub.mk
#   Author : Jeroen
#   Date   : Sept 09 2004

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/Services

SRCS     += 	$(SRCDIR)/Service.cc \
		$(SRCDIR)/ServiceDB.cc \
		$(SRCDIR)/ServiceFrame.cc \
		$(SRCDIR)/ServiceLog.cc \
		$(SRCDIR)/ServiceManager.cc \
		$(SRCDIR)/ServiceNode.cc \
		$(SRCDIR)/ServiceClient.cc \
		$(SRCDIR)/SimpleService.cc \
		$(SRCDIR)/FileTransfer.cc \
		$(SRCDIR)/FileTransferClient.cc
			
PSELIBS := \
	Core/Thread \
	Core/Util \
	Core/Exceptions \
	Core/Containers \
	Core/ICom \
	Core/SystemCall \
	Dataflow/XMLUtil \
	Dataflow/Network \

LIBS := $(XML_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
