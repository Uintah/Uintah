#
#  The MIT License
#
#  Copyright (c) 1997-2015 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
###################################################################
#
#  Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Core/Util

SRCS += \
        $(SRCDIR)/DebugStream.cc        \
        $(SRCDIR)/Endian.cc             \
        $(SRCDIR)/Environment.cc        \
        $(SRCDIR)/FileUtils.cc          \
        $(SRCDIR)/SizeTypeConvert.cc    \
        $(SRCDIR)/RWS.cc                \
        $(SRCDIR)/sci_system.cc         \
        $(SRCDIR)/Signals.cc            \
        $(SRCDIR)/Timer.cc              \
        $(SRCDIR)/TypeDescription.cc    \
        $(SRCDIR)/ProgressiveWarning.cc \
        $(SRCDIR)/ProgressReporter.cc   \
        $(SRCDIR)/Util.cc               \
        $(SRCDIR)/DynamicLoader.cc      \
        $(SRCDIR)/soloader.cc           \
        $(SRCDIR)/Socket.cc  

SRCS += $(REFCOUNT_IMPL)

ifeq ($(HAVE_CUDA),yes)
  SRCS += $(SRCDIR)/GPU.cu
  DLINK_FILES += Core/Util/GPU.o
endif

PSELIBS := Core/Containers Core/Exceptions Core/Thread Core/Malloc

LIBS := $(DL_LIBRARY) $(THREAD_LIBRARY) $(SOCKET_LIBRARY) $(CUDA_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
