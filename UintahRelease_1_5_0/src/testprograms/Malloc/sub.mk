#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the \"Software\"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 
# 
# 
# 
# Makefile fragment for this subdirectory 

SRCDIR := testprograms/Malloc

PSELIBS :=
LIBS := $(XML_LIBRARY) 

PROGRAM := $(SRCDIR)/test1
SRCS := $(SRCDIR)/test1.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test2
SRCS := $(SRCDIR)/test2.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test3
SRCS := $(SRCDIR)/test3.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test4
SRCS := $(SRCDIR)/test4.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test5
SRCS := $(SRCDIR)/test5.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test6
SRCS := $(SRCDIR)/test6.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test8
SRCS := $(SRCDIR)/test8.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test9
SRCS := $(SRCDIR)/test9.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test10
SRCS := $(SRCDIR)/test10.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test11
SRCS := $(SRCDIR)/test11.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test12
SRCS := $(SRCDIR)/test12.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test13
SRCS := $(SRCDIR)/test13.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test14
SRCS := $(SRCDIR)/test14.cc
include $(SCIRUN_SCRIPTS)/program.mk

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := $(CORE_STATIC_PSELIBS)
else # Non-static build
  PSELIBS := Core/Util Core/Containers Core/Exceptions Core/Thread
endif

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := $(CORE_STATIC_LIBS)
else
  LIBS := $(XML_LIBRARY) $(DL_LIBRARY) $(THREAD_LIBRARY) 
endif

PROGRAM := $(SRCDIR)/test15
SRCS := $(SRCDIR)/test15.cc

include $(SCIRUN_SCRIPTS)/program.mk

PSELIBS :=

