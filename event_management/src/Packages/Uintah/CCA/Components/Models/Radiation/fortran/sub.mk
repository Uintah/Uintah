# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/Models/Radiation/fortran

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

$(SRCDIR)/m_cellg.o: $(SRCDIR)/m_cellg_fort.h
$(SRCDIR)/m_radarray.o: $(SRCDIR)/m_radarray_fort.h
$(SRCDIR)/m_radcal.o: $(SRCDIR)/m_radcal_fort.h
$(SRCDIR)/m_radcoef.o: $(SRCDIR)/m_radcoef_fort.h
$(SRCDIR)/m_radwsgg.o: $(SRCDIR)/m_radwsgg_fort.h
$(SRCDIR)/m_rordr.o: $(SRCDIR)/m_rordr_fort.h
$(SRCDIR)/m_rordrss.o: $(SRCDIR)/m_rordrss_fort.h
$(SRCDIR)/m_rordrtn.o: $(SRCDIR)/m_rordrtn_fort.h
$(SRCDIR)/m_rdombc.o: $(SRCDIR)/m_rdombc_fort.h
$(SRCDIR)/m_rdombmcalc.o: $(SRCDIR)/m_rdombmcalc_fort.h
$(SRCDIR)/m_rdomsolve.o: $(SRCDIR)/m_rdomsolve_fort.h
$(SRCDIR)/m_rdomsrc.o: $(SRCDIR)/m_rdomsrc_fort.h
$(SRCDIR)/m_rdomflux.o: $(SRCDIR)/m_rdomflux_fort.h
$(SRCDIR)/m_rdomvolq.o: $(SRCDIR)/m_rdomvolq_fort.h
$(SRCDIR)/m_rshsolve.o: $(SRCDIR)/m_rshsolve_fort.h
$(SRCDIR)/m_rshresults.o: $(SRCDIR)/m_rshresults_fort.h

