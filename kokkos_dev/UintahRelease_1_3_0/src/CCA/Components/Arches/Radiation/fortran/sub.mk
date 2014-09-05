# 
# 
# The MIT License
# 
# Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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

SRCDIR   := CCA/Components/Arches/Radiation/fortran

SRCS += \
	$(SRCDIR)/eco.F \
	$(SRCDIR)/eco2.F \
	$(SRCDIR)/eh2o.F \
	$(SRCDIR)/efuel.F \
	$(SRCDIR)/eico2.F \
	$(SRCDIR)/eih2o.F \
	$(SRCDIR)/find.F \
	$(SRCDIR)/fixradval.F \
	$(SRCDIR)/radarray.F \
	$(SRCDIR)/radcoef.F \
	$(SRCDIR)/radcal.F \
	$(SRCDIR)/radwsgg.F \
	$(SRCDIR)/rdombc.F \
	$(SRCDIR)/rdombmcalc.F \
	$(SRCDIR)/rdomflux.F \
	$(SRCDIR)/rdomsolve.F \
	$(SRCDIR)/rdomsrc.F \
	$(SRCDIR)/rdomvolq.F \
	$(SRCDIR)/rordr.F \
	$(SRCDIR)/rordrss.F \
	$(SRCDIR)/rordrtn.F \
	$(SRCDIR)/rshsolve.F \
	$(SRCDIR)/rshresults.F \
	$(SRCDIR)/soot.F

PSELIBS := 

LIBS := $(F_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/radarray.$(OBJEXT): $(SRCDIR)/radarray_fort.h
$(SRCDIR)/radcal.$(OBJEXT): $(SRCDIR)/radcal_fort.h
$(SRCDIR)/radcoef.$(OBJEXT): $(SRCDIR)/radcoef_fort.h
$(SRCDIR)/radwsgg.$(OBJEXT): $(SRCDIR)/radwsgg_fort.h
$(SRCDIR)/rordr.$(OBJEXT): $(SRCDIR)/rordr_fort.h
$(SRCDIR)/rordrss.$(OBJEXT): $(SRCDIR)/rordrss_fort.h
$(SRCDIR)/rordrtn.$(OBJEXT): $(SRCDIR)/rordrtn_fort.h
$(SRCDIR)/rdombc.$(OBJEXT): $(SRCDIR)/rdombc_fort.h
$(SRCDIR)/rdombmcalc.$(OBJEXT): $(SRCDIR)/rdombmcalc_fort.h
$(SRCDIR)/rdomsolve.$(OBJEXT): $(SRCDIR)/rdomsolve_fort.h
$(SRCDIR)/rdomsrc.$(OBJEXT): $(SRCDIR)/rdomsrc_fort.h
$(SRCDIR)/rdomflux.$(OBJEXT): $(SRCDIR)/rdomflux_fort.h
$(SRCDIR)/rdomvolq.$(OBJEXT): $(SRCDIR)/rdomvolq_fort.h
$(SRCDIR)/rshsolve.$(OBJEXT): $(SRCDIR)/rshsolve_fort.h
$(SRCDIR)/rshresults.$(OBJEXT): $(SRCDIR)/rshresults_fort.h

