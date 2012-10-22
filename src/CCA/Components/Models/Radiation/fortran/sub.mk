#
#  The MIT License
#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

SRCDIR   := CCA/Components/Models/Radiation/fortran

SRCS += \
        $(SRCDIR)/m_cellg.F \
        $(SRCDIR)/m_eco.F \
        $(SRCDIR)/m_eco2.F \
        $(SRCDIR)/m_eh2o.F \
        $(SRCDIR)/m_efuel.F \
        $(SRCDIR)/m_eico2.F \
        $(SRCDIR)/m_eih2o.F \
        $(SRCDIR)/m_find.F \
        $(SRCDIR)/m_fixradval.F \
        $(SRCDIR)/m_radarray.F \
        $(SRCDIR)/m_radcoef.F \
        $(SRCDIR)/m_radcal.F \
        $(SRCDIR)/m_radwsgg.F \
        $(SRCDIR)/m_rdombc.F \
        $(SRCDIR)/m_rdombmcalc.F \
        $(SRCDIR)/m_rdomflux.F \
        $(SRCDIR)/m_rdomsolve.F \
        $(SRCDIR)/m_rdomsrc.F \
        $(SRCDIR)/m_rdomvolq.F \
        $(SRCDIR)/m_rordr.F \
        $(SRCDIR)/m_rordrss.F \
        $(SRCDIR)/m_rordrtn.F \
        $(SRCDIR)/m_rshsolve.F \
        $(SRCDIR)/m_rshresults.F \
        $(SRCDIR)/m_soot.F

$(SRCDIR)/m_cellg.$(OBJEXT): $(SRCDIR)/m_cellg_fort.h
$(SRCDIR)/m_radarray.$(OBJEXT): $(SRCDIR)/m_radarray_fort.h
$(SRCDIR)/m_radcal.$(OBJEXT): $(SRCDIR)/m_radcal_fort.h
$(SRCDIR)/m_radcoef.$(OBJEXT): $(SRCDIR)/m_radcoef_fort.h
$(SRCDIR)/m_radwsgg.$(OBJEXT): $(SRCDIR)/m_radwsgg_fort.h
$(SRCDIR)/m_rordr.$(OBJEXT): $(SRCDIR)/m_rordr_fort.h
$(SRCDIR)/m_rordrss.$(OBJEXT): $(SRCDIR)/m_rordrss_fort.h
$(SRCDIR)/m_rordrtn.$(OBJEXT): $(SRCDIR)/m_rordrtn_fort.h
$(SRCDIR)/m_rdombc.$(OBJEXT): $(SRCDIR)/m_rdombc_fort.h
$(SRCDIR)/m_rdombmcalc.$(OBJEXT): $(SRCDIR)/m_rdombmcalc_fort.h
$(SRCDIR)/m_rdomsolve.$(OBJEXT): $(SRCDIR)/m_rdomsolve_fort.h
$(SRCDIR)/m_rdomsrc.$(OBJEXT): $(SRCDIR)/m_rdomsrc_fort.h
$(SRCDIR)/m_rdomflux.$(OBJEXT): $(SRCDIR)/m_rdomflux_fort.h
$(SRCDIR)/m_rdomvolq.$(OBJEXT): $(SRCDIR)/m_rdomvolq_fort.h
$(SRCDIR)/m_rshsolve.$(OBJEXT): $(SRCDIR)/m_rshsolve_fort.h
$(SRCDIR)/m_rshresults.$(OBJEXT): $(SRCDIR)/m_rshresults_fort.h

