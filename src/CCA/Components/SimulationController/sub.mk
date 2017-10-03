#
#  The MIT License
#
#  Copyright (c) 1997-2017 The University of Utah
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

SRCDIR   := CCA/Components/SimulationController

SRCS     += $(SRCDIR)/SimulationController.cc \
            $(SRCDIR)/AMRSimulationController.cc 

PSELIBS := \
	Core/DataArchive \
	Core/Disclosure  \
	Core/Exceptions  \
	Core/Grid        \
	Core/Parallel    \
	Core/ProblemSpec \
	Core/Util        \
	CCA/Components/DataArchiver    \
	CCA/Components/ReduceUda       \
	CCA/Components/Regridder       \
	CCA/Components/Schedulers      \
	CCA/Ports        \
	Core/OS       \
	Core/Geometry \
	Core/Util     \
	Core/Exceptions \
	\
	Core/Math

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(PAPI_LIBRARY) $(M_LIBRARY) $(GPERFTOOLS_LIBRARY)

ifeq ($(HAVE_VISIT),yes)
  INCLUDES += $(VISIT_INCLUDE)
  PSELIBS += VisIt/libsim
  LIBS += $(VISIT_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

