# 
# 
# The MIT License
# 
# Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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

SRCDIR  := Packages/Uintah/StandAlone/tools/dumpfields
#PROGRAM := Packages/Uintah/StandAlone/dumpfields
PROGRAM := Packages/Uintah/StandAlone/tools/dumpfields/dumpfields

SRCS    := \
	$(SRCDIR)/dumpfields.cc \
	\
	$(SRCDIR)/utils.h $(SRCDIR)/utils.cc \
	$(SRCDIR)/Args.h  $(SRCDIR)/Args.cc \
	$(SRCDIR)/FieldSelection.h $(SRCDIR)/FieldSelection.cc \
	\
	$(SRCDIR)/FieldDiags.h $(SRCDIR)/FieldDiags.cc \
	$(SRCDIR)/ScalarDiags.h $(SRCDIR)/ScalarDiags.cc \
	$(SRCDIR)/VectorDiags.h $(SRCDIR)/VectorDiags.cc \
	$(SRCDIR)/TensorDiags.h $(SRCDIR)/TensorDiags.cc \
	\
	$(SRCDIR)/FieldDumper.h $(SRCDIR)/FieldDumper.cc \
	$(SRCDIR)/TextDumper.h $(SRCDIR)/TextDumper.cc \
	$(SRCDIR)/EnsightDumper.h $(SRCDIR)/EnsightDumper.cc \
	$(SRCDIR)/InfoDumper.h $(SRCDIR)/InfoDumper.cc \
	$(SRCDIR)/HistogramDumper.h $(SRCDIR)/HistogramDumper.cc 

ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/Uintah
else
  PSELIBS := \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/DataArchive   \
	Packages/Uintah/CCA/Ports          \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/XMLUtil \
        Core/Exceptions  \
        Core/Persistent  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

