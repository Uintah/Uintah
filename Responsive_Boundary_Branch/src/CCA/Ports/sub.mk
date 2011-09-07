# 
# 
# The MIT License
# 
# Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 
# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Ports

SRCS += $(SRCDIR)/SimulationInterface.cc \
	$(SRCDIR)/DataWarehouse.cc \
	$(SRCDIR)/LoadBalancer.cc \
	$(SRCDIR)/ModelInterface.cc \
	$(SRCDIR)/ModelMaker.cc \
	$(SRCDIR)/Output.cc \
	$(SRCDIR)/ProblemSpecInterface.cc \
	$(SRCDIR)/Regridder.cc \
	$(SRCDIR)/Scheduler.cc \
	$(SRCDIR)/SolverInterface.cc \
	$(SRCDIR)/SwitchingCriteria.cc \
	$(SRCDIR)/SFC.cc

PSELIBS := \
	Core/Parallel    \
	Core/Util        \
	Core/Grid        \
	Core/Disclosure  \
	Core/Exceptions  \
	Core/ProblemSpec \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Containers                  \
	Core/Util

LIBS := $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

