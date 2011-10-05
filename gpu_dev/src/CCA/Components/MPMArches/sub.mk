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

SRCDIR   := CCA/Components/MPMArches

SRCS     += \
	$(SRCDIR)/MPMArches.cc \
	$(SRCDIR)/MPMArchesLabel.cc \
	$(SRCDIR)/CutCellInfo.cc

SUBDIRS := $(SRCDIR)/fortran 
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	CCA/Ports          \
	Core/Grid          \
	Core/Util          \
	Core/Labels        \
	Core/Disclosure    \
	Core/Parallel      \
	Core/ProblemSpec   \
	Core/Exceptions    \
	Core/Math          \
	CCA/Components/MPM \
	CCA/Components/Arches \
	CCA/Components/Arches/fortran \
	CCA/Components/Arches/Mixing  \
       CCA/Components/OnTheFlyAnalysis \
	Core/Exceptions \
	Core/Util       \
	Core/Thread     \
	Core/Geometry   

LIBS := $(XML2_LIBRARY) $(PETSC_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/MPMArches.$(OBJEXT): $(SRCDIR)/fortran/collect_drag_cc_fort.h
$(SRCDIR)/MPMArches.$(OBJEXT): $(SRCDIR)/fortran/collect_scalar_fctocc_fort.h
$(SRCDIR)/MPMArches.$(OBJEXT): $(SRCDIR)/fortran/energy_exchange_term_fort.h
$(SRCDIR)/MPMArches.$(OBJEXT): $(SRCDIR)/fortran/interp_centertoface_fort.h
$(SRCDIR)/MPMArches.$(OBJEXT): $(SRCDIR)/fortran/momentum_exchange_term_continuous_cc_fort.h
$(SRCDIR)/MPMArches.$(OBJEXT): $(SRCDIR)/fortran/pressure_force_fort.h
$(SRCDIR)/MPMArches.$(OBJEXT): $(SRCDIR)/fortran/read_complex_geometry_fort.h
$(SRCDIR)/MPMArches.$(OBJEXT): $(SRCDIR)/fortran/read_complex_geometry_walls_fort.h
