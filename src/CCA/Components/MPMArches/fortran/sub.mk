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

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/MPMArches/fortran

SRCS += $(SRCDIR)/collect_drag_cc.F \
	$(SRCDIR)/collect_scalar_fctocc.F \
	$(SRCDIR)/energy_exchange_term.F \
	$(SRCDIR)/interp_centertoface.F \
	$(SRCDIR)/momentum_exchange_term_continuous_cc.F \
	$(SRCDIR)/pressure_force.F \
	$(SRCDIR)/read_complex_geometry.F \
	$(SRCDIR)/read_complex_geometry_walls.F \
	$(SRCDIR)/taucal_cc.F \
	$(SRCDIR)/walmom_cc.F

#include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/collect_drag_cc.$(OBJEXT): $(SRCDIR)/collect_drag_cc_fort.h
$(SRCDIR)/collect_scalar_fctocc.$(OBJEXT): $(SRCDIR)/collect_scalar_fctocc_fort.h
$(SRCDIR)/energy_exchange_term.$(OBJEXT): $(SRCDIR)/energy_exchange_term_fort.h
$(SRCDIR)/interp_centertoface.$(OBJEXT): $(SRCDIR)/interp_centertoface_fort.h
$(SRCDIR)/momentum_exchange_term_continuous_cc.$(OBJEXT): $(SRCDIR)/momentum_exchange_term_continuous_cc_fort.h
$(SRCDIR)/pressure_force.$(OBJEXT): $(SRCDIR)/pressure_force_fort.h
$(SRCDIR)/read_complex_geometry.$(OBJEXT): $(SRCDIR)/read_complex_geometry_fort.h
$(SRCDIR)/read_complex_geometry_walls.$(OBJEXT): $(SRCDIR)/read_complex_geometry_walls_fort.h
$(SRCDIR)/taucal_cc.$(OBJEXT): $(SRCDIR)/taucal_cc_fort.h
$(SRCDIR)/walmom_cc.$(OBJEXT): $(SRCDIR)/walmom_cc_fort.h
