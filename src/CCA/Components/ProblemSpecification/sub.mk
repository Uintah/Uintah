#
#  The MIT License
#
#  Copyright (c) 1997-2025 The University of Utah
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
# 
# 
# 
# 
# Makefile fragment for this subdirectory 


include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/ProblemSpecification

SRCS    += $(SRCDIR)/ProblemSpecReader.cc

PSELIBS := \
        CCA/Ports        \
        Core/Containers  \
        Core/Exceptions  \
        Core/Grid        \
        Core/Parallel    \
        Core/ProblemSpec \
        Core/Util        

LIBS := $(XML2_LIBRARY)

# See commit message for while hacking in the MPI_LIBRARY is necessary.
ifeq ($(IS_OSX),yes)
  LIBS += $(MPI_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

#PROGRAM        := $(SRCDIR)/testing
#SRCS   := $(SRCDIR)/testing.cc
#include $(SCIRUN_SCRIPTS)/program.mk
#PROGRAM        := $(SRCDIR)/test2
#SRCS   := $(SRCDIR)/test2.cc 
#ifeq ($(LARGESOS),yes)
#  PSELIBS := Packages/Uintah
#else
#  PSELIBS := \
#       CCA/Ports \
#       Core/Grid \
#       Core/ProblemSpec \
#       CCA/Components/ProblemSpecification \
#endif
#LIBS   := $(XML2_LIBRARY)
#include $(SCIRUN_SCRIPTS)/program.mk


