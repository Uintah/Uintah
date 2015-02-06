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

SRCDIR := testprograms/Thread

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := $(CORE_STATIC_PSELIBS)
else # Non-static build
  PSELIBS := Core/Thread Core/Util
endif

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := $(CORE_STATIC_LIBS)
else
  LIBS := $(THREAD_LIBRARY) $(XML_LIBRARY)
endif

PROGRAM := $(SRCDIR)/bps
SRCS := $(SRCDIR)/bps.cc

include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/threadexit
SRCS := $(SRCDIR)/threadexit.cc

include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/parallel
SRCS := $(SRCDIR)/parallel.cc

include $(SCIRUN_SCRIPTS)/program.mk

