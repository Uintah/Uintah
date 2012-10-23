#
#  The MIT License
#
#  Copyright (c) 1997-2012 The University of Utah
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

SRCDIR  := StandAlone/tools/tracker
PROGRAM := $(SRCDIR)/tracker

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := Core/Tracker $(CORE_STATIC_PSELIBS)
else # Non-static build
  ifeq ($(LARGESOS),yes)
    PSELIBS := Datflow Packages/Uintah
  else
    PSELIBS := \
	  Core/Util                     \
          Core/Tracker
  endif
endif

SRCS := \
	$(SRCDIR)/TrackerProgram.cc                  

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := $(CORE_STATIC_LIBS)
else
  LIBS := $(XML2_LIBRARY) $(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

